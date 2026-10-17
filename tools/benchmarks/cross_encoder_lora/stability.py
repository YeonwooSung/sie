"""Baseline-only stability diagnostic; never qualifies a candidate or provisions compute.

Run with --inputs, --fixtures and --output. Six fresh baseline processes each
execute C1/C4/C16 with fixed S/L1/L2 windows: 30/200/200 sequential warmup
requests followed by the existing 200-request/12-second measurement. Later
window agreement is tested symmetrically; between-process dispersion remains
descriptive. The original timed methods and automatic GC behavior are reused.
"""

# Only this checkout's Python worker runs; no shell or allocation.
# ruff: noqa: S603

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import resource
import signal
import socket
import statistics
import subprocess
import sys
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

import torch
import uvicorn

from tools.benchmarks.cross_encoder_lora import worker
from tools.benchmarks.cross_encoder_lora.adapter import require_forward_selection
from tools.benchmarks.cross_encoder_lora.evidence import Evidence, atomic_json, digest, fingerprint, vector_delta
from tools.benchmarks.cross_encoder_lora.protocol import (
    BASE_REVISION,
    CPU_ORACLE,
    MAX_BATCH_ITEMS,
    MODEL,
    ORACLE_ATOL,
    RESTORE_ATOL,
    SEQUENTIAL,
    STEADY_SECONDS,
    THREADS,
    WEIGHTS_SHA256,
    paired_gate,
    percentile,
    require,
)
from tools.benchmarks.cross_encoder_lora.runner import (
    KILL_GRACE_SECONDS,
    TERMINATE_GRACE_SECONDS,
    WAIT_GRACE_SECONDS,
    extract,
    harness_manifest,
    source_environment,
    stop,
    terminate,
)

PROCESSES = 6
WINDOWS = (("S", 30), ("L1", 200), ("L2", 200))
CONCURRENCIES = (1, 4, 16)
METRICS = ((1, "mean_ms"), (1, "p95_ms"), (4, "rps"), (4, "p95_ms"), (16, "rps"), (16, "p95_ms"))
RUNNER_LIMIT = 1200
WORKER_LIMIT = 180
GC_FIELDS = ("collections", "collected", "uncollectable")
RUSAGE_FIELDS = ("ru_utime", "ru_stime", "ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw")
OPTIONAL_FILES = ("cpu.stat", "cpu.stat.local", "cpu.pressure", "cpu.max", "cpu.weight", "cpuset.cpus.effective")
COUNTER_FILES = ("cpu.stat", "cpu.stat.local")
CONTROL_FILES = ("cpu.max", "cpu.weight", "cpuset.cpus.effective")
GC_GENERATIONS = 3
QUARTILES = 4
PSI_FIELDS = ("avg10", "avg60", "avg300", "total")
MAX_PERCENT = 100
MAX_CPU_WEIGHT = 10000


class DiagnosticEvidence(Evidence):
    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        if fields.get("qualification"):
            fields["hardware_preflight_only"] = True
        fields["qualification"] = False
        return super().emit(event, **fields)


def unified_cgroup(proc: Path = Path("/proc/self")) -> dict[str, Any]:
    """Resolve the process's own namespace; never fall back to another cgroup."""
    try:
        rows = (proc / "cgroup").read_text().splitlines()
        membership = [row[3:] for row in rows if row.startswith("0::")]
        require(len(membership) == 1, "unified cgroup membership is unavailable or ambiguous")
        group = Path(membership[0])
        require(group.is_absolute() and ".." not in group.parts, "invalid cgroup membership")
        matches = []
        for row in (proc / "mountinfo").read_text().splitlines():
            left, right = row.split(" - ", 1)
            if right.split()[0] != "cgroup2":
                continue
            fields = left.split()
            root = Path(fields[3].replace("\\040", " ").replace("\\134", "\\"))
            mount = Path(fields[4].replace("\\040", " ").replace("\\134", "\\"))
            if group.is_relative_to(root):
                matches.append(mount / group.relative_to(root))
        require(len(matches) == 1, "cgroup mount cannot be resolved uniquely")
        path = matches[0]
        info = path.stat()
        return {"path": str(path), "device": info.st_dev, "inode": info.st_ino, "error": None}
    except (OSError, ValueError, IndexError, AssertionError) as error:
        return {"path": None, "error": f"{type(error).__name__}: {error}"}


def flat_counters(text: str) -> dict[str, int]:
    result = {}
    for row in text.splitlines():
        key, value = row.split()
        require(key not in result, "duplicate telemetry counter")
        result[key] = int(value)
        require(result[key] >= 0, "negative telemetry counter")
    require(bool(result), "empty exposed telemetry counter file")
    return result


def pressure(text: str) -> dict[str, dict[str, float | int]]:
    result = {}
    for row in text.splitlines():
        name, *parts = row.split()
        require(name in {"some", "full"} and name not in result, "invalid PSI row")
        values = dict(part.split("=", 1) for part in parts)
        require(len(parts) == len(PSI_FIELDS) and set(values) == set(PSI_FIELDS), "invalid PSI fields")
        parsed = {key: int(value) if key == "total" else float(value) for key, value in values.items()}
        require(all(math.isfinite(value) and value >= 0 for value in parsed.values()), "invalid PSI value")
        require(all(parsed[key] <= MAX_PERCENT for key in PSI_FIELDS[:-1]), "invalid PSI percentage")
        result[name] = parsed
    require("some" in result, "missing exposed PSI some row")
    return result


def validate_core(core: dict[str, Any]) -> None:
    require(
        all(isinstance(core[key], int) and core[key] > 0 for key in ("wall_ns", "monotonic_ns"))
        and core["gc_enabled"] is True,
        "invalid required clock/GC state",
    )
    require(
        set(core["rusage"]) == set(RUSAGE_FIELDS)
        and all(math.isfinite(value) and value >= 0 for value in core["rusage"].values()),
        "invalid required rusage",
    )
    for key in ("gc_thresholds", "gc_count"):
        require(
            len(core[key]) == GC_GENERATIONS and all(type(value) is int and value >= 0 for value in core[key]),
            "invalid required GC counts/thresholds",
        )
    require(core["gc_thresholds"][0] > 0, "automatic GC is disabled by its threshold")
    require(len(core["gc_stats"]) == GC_GENERATIONS, "missing required GC generations")
    require(
        all(
            set(row) == set(GC_FIELDS) and all(type(row[key]) is int and row[key] >= 0 for key in GC_FIELDS)
            for row in core["gc_stats"]
        ),
        "invalid required GC statistics",
    )


def control_value(name: str, text: str) -> str:
    if name == "cpu.max":
        quota, period = text.split()
        require((quota == "max" or int(quota) > 0) and int(period) > 0, "invalid exposed cpu.max")
    elif name == "cpu.weight":
        require(1 <= int(text) <= MAX_CPU_WEIGHT, "invalid exposed cpu.weight")
    else:
        for item in text.split(","):
            bounds = [int(value) for value in item.split("-")]
            require(len(bounds) in {1, 2} and bounds[0] >= 0 and bounds[-1] >= bounds[0], "invalid exposed cpuset")
    return text


def snapshot() -> dict[str, Any]:
    """Only called outside performance(), including all filesystem reads."""
    started = time.monotonic_ns()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    core = {
        "wall_ns": time.time_ns(),
        "monotonic_ns": started,
        "rusage": {key: getattr(usage, key) for key in RUSAGE_FIELDS},
        "gc_enabled": gc.isenabled(),
        "gc_thresholds": list(gc.get_threshold()),
        "gc_count": list(gc.get_count()),
        "gc_stats": gc.get_stats(),
    }
    validate_core(core)
    group = unified_cgroup()
    files: dict[str, Any] = {}
    errors = {}
    if group["path"] is not None:
        for name in OPTIONAL_FILES:
            try:
                text = (Path(group["path"]) / name).read_text().strip()
            except OSError as error:
                files[name] = None
                errors[name] = f"{type(error).__name__}: {error}"
                continue
            # Missing optional files are unavailability, but malformed exposed
            # counters must not be mistaken for trustworthy zero deltas.
            files[name] = (
                flat_counters(text)
                if name in COUNTER_FILES
                else pressure(text)
                if name == "cpu.pressure"
                else control_value(name, text)
            )
    else:
        files = dict.fromkeys(OPTIONAL_FILES)
        errors["cgroup"] = group["error"]
    affinity = None
    try:
        if hasattr(os, "sched_getaffinity"):
            affinity = sorted(os.sched_getaffinity(0))
            require(bool(affinity), "empty exposed affinity")
        else:
            errors["affinity"] = "sched_getaffinity unavailable on this platform"
    except OSError as error:
        errors["affinity"] = f"{type(error).__name__}: {error}"
    return {
        "qualification": False,
        "core": core,
        "cgroup": group,
        "cgroup_files": files,
        "optional_errors": errors,
        "affinity": affinity,
        "snapshot_read_ns": time.monotonic_ns() - started,
    }


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    require(set(before) == set(after), "counter fields changed within a window")
    result = {key: after[key] - value for key, value in before.items()}
    require(all(math.isfinite(value) and value >= 0 for value in result.values()), "telemetry counter reset")
    return result


def telemetry_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    first, last = before["core"], after["core"]
    validate_core(first)
    validate_core(last)
    require(
        last["wall_ns"] > first["wall_ns"] and last["monotonic_ns"] > first["monotonic_ns"],
        "required clock did not advance",
    )
    require(
        first["gc_enabled"] and last["gc_enabled"] and first["gc_thresholds"] == last["gc_thresholds"],
        "GC configuration changed",
    )
    group_before, group_after = before["cgroup"], after["cgroup"]
    require(
        all(group_before.get(key) == group_after.get(key) for key in ("path", "device", "inode")),
        "cgroup path or identity changed within a window",
    )
    require(before["affinity"] == after["affinity"], "process affinity changed within a window")
    for name in CONTROL_FILES:
        require(
            before["cgroup_files"].get(name) == after["cgroup_files"].get(name),
            "cgroup quota/weight/cpuset changed within a window",
        )
    optional = {}
    for name in COUNTER_FILES:
        a, b = before["cgroup_files"].get(name), after["cgroup_files"].get(name)
        optional[name] = counter_delta(a, b) if a is not None and b is not None else None
    a, b = before["cgroup_files"].get("cpu.pressure"), after["cgroup_files"].get("cpu.pressure")
    optional["cpu.pressure"] = (
        counter_delta({k: v["total"] for k, v in a.items()}, {k: v["total"] for k, v in b.items()})
        if a is not None and b is not None
        else None
    )
    return {
        "qualification": False,
        "scope": "Whole performance call, including sequential warmup, measurement, drain, validation and raw emission",
        "wall_s": (last["wall_ns"] - first["wall_ns"]) / 1e9,
        "monotonic_s": (last["monotonic_ns"] - first["monotonic_ns"]) / 1e9,
        "rusage": counter_delta(first["rusage"], last["rusage"]),
        "gc_stats": [counter_delta(a, b) for a, b in zip(first["gc_stats"], last["gc_stats"], strict=True)],
        "optional_cgroup": optional,
        "cpu_pressure_counters_available": all(optional.get(name) is not None for name in ("cpu.stat", "cpu.pressure")),
        "resource_telemetry_complete": before["affinity"] is not None
        and after["affinity"] is not None
        and all(
            before["cgroup_files"].get(name) is not None and after["cgroup_files"].get(name) is not None
            for name in OPTIONAL_FILES
        ),
        "optional_errors_before": before["optional_errors"],
        "optional_errors_after": after["optional_errors"],
    }


def metrics(phase: dict[str, Any]) -> dict[str, Any]:
    samples = phase["samples"]
    latencies = [row["latency_ms"] for row in samples]
    require(bool(latencies) and all(math.isfinite(value) and value > 0 for value in latencies), "invalid raw latency")
    elapsed = phase["elapsed_s"]
    require(math.isfinite(elapsed) and elapsed > 0, "invalid phase elapsed")
    result = {
        "count": len(samples),
        "elapsed_s": elapsed,
        "mean_ms": statistics.mean(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "rps": len(samples) / elapsed,
    }
    for key, value in result.items():
        require(math.isclose(value, phase[key], rel_tol=1e-12, abs_tol=0), "raw metric differs from phase aggregate")
    return result


def chronological_quartiles(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(samples, key=lambda row: (row["start_ns"], row["label"]))
    require(len(ordered) >= QUARTILES, "too few requests for fixed chronological quartiles")
    result = []
    for index in range(QUARTILES):
        rows = ordered[len(ordered) * index // QUARTILES : len(ordered) * (index + 1) // QUARTILES]
        headers: dict[str, Any] = {}
        keys = sorted({key for row in rows for key in row["responses"][0]["timing_headers"]})
        for key in keys:
            values = [
                float(row["responses"][0]["timing_headers"][key])
                for row in rows
                if key in row["responses"][0]["timing_headers"]
            ]
            require(all(math.isfinite(value) and value >= 0 for value in values), "invalid timing header")
            headers[key] = {
                "count": len(values),
                "min": min(values),
                "mean": statistics.mean(values),
                "max": max(values),
                "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1],
            }
        latencies = [row["latency_ms"] for row in rows]
        result.append(
            {
                "count": len(rows),
                "start_ns": rows[0]["start_ns"],
                "last_start_ns": rows[-1]["start_ns"],
                "mean_ms": statistics.mean(latencies),
                "p95_ms": percentile(latencies, 0.95),
                "timing_headers": headers,
            }
        )
    return result


def verify_attempt(path: Path, record: dict[str, Any]) -> None:
    """Reopen raw warmups/statuses as well as measured rows; never time this work."""
    events = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(events) == 1 and events[0]["event"] == "performance_attempt", "missing/extra performance attempt")
    attempt = events[0]
    require(
        attempt["qualification"] is False and attempt["concurrency"] == record["concurrency"],
        "attempt identity differs",
    )
    require(len(attempt["warmups"]) == record["warmups"], "sequential warmup count differs")
    measured = record["performance"]
    require(
        attempt["completed_samples"]
        == [
            {key: value for key, value in row.items() if key != "diagnostic_matching_oracle"}
            for row in measured["samples"]
        ],
        "attempt/raw measured rows differ",
    )
    rows = [*attempt["warmups"], *attempt["completed_samples"]]
    require(
        len(attempt["wire_responses"]) == len(rows)
        and all(row["status"] == HTTPStatus.OK for row in attempt["wire_responses"]),
        "hidden wire retries or failure",
    )
    for row in rows:
        saved = row["scores"][:]
        worker.validate_response(row)
        require(saved == row["scores"] and row["selection"] is None, "baseline scores/selection differ")
        require(row["end_ns"] >= row["start_ns"] > 0, "invalid request timestamps")
    if record["concurrency"] == 1:
        require(measured["count"] == (4 if record["smoke"] else SEQUENTIAL), "sequential request count differs")
    else:
        require(measured["elapsed_s"] >= (0.15 if record["smoke"] else STEADY_SECONDS), "closed-loop duration differs")


def measure_window(
    url: str,
    concurrency: int,
    window: str,
    warmups: int,
    directory: Path,
    base_scores: list[float],
    oracles: dict[str, Any],
    *,
    smoke: bool,
    continuity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = DiagnosticEvidence(directory)
    record: dict[str, Any] = {
        "qualification": False,
        "completed": False,
        "window": window,
        "concurrency": concurrency,
        "warmups": 2 if smoke else warmups,
        "smoke": smoke,
    }
    original_warmups = worker.WARMUP
    try:
        record["gpu_before"] = worker.telemetry()
        record["telemetry_before"] = snapshot()
        if continuity is not None:
            telemetry_delta(continuity, record["telemetry_before"])
        worker.WARMUP = warmups  # ty: ignore[invalid-assignment] -- scoped diagnostic value, always restored
        try:
            measured = worker.performance(url, concurrency, evidence, smoke=smoke)
        finally:
            worker.WARMUP = original_warmups
        record["telemetry_after"] = snapshot()
        record["gpu_after"] = worker.telemetry()
        record["telemetry_delta"] = telemetry_delta(record["telemetry_before"], record["telemetry_after"])
        record["performance"] = measured
        record["attempt_sha256"] = digest(directory / "events.jsonl")
        verify_attempt(directory / "events.jsonl", record)
        for row in measured["samples"]:
            if concurrency == 1:
                require(vector_delta(row["scores"], base_scores) <= RESTORE_ATOL, "baseline C1 scores changed")
            row["diagnostic_matching_oracle"] = worker.matching_vector(row["scores"], oracles)
        record["metrics"] = metrics(measured)
        record["chronological_quartiles"] = chronological_quartiles(measured["samples"])
        record["actual_timed_batch_shape"] = "unknown"
        record["completed"] = True
        if continuity is not None:
            continuity.clear()
            continuity.update(record["telemetry_after"])
    except BaseException as error:
        record["error"] = worker.error_record(error)
        raise
    finally:
        worker.WARMUP = original_warmups
        atomic_json(directory / "window.json", record)
    return {
        "qualification": False,
        "window": window,
        "concurrency": concurrency,
        "completed": True,
        "metrics": record["metrics"],
        "resource_telemetry_complete": record["telemetry_delta"]["resource_telemetry_complete"],
        "record": str((directory / "window.json").relative_to(directory.parent.parent)),
        "sha256": digest(directory / "window.json"),
    }


async def serve_windows(args: argparse.Namespace, evidence: DiagnosticEvidence, summary: dict[str, Any]) -> None:
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    require(bool(manifest["smoke"]) == args.smoke, "fixture mode mismatch")
    snapshot_path = Path(manifest["snapshot"])
    for name, expected in manifest["model_files"].items():
        require(digest(snapshot_path / name) == expected, "model fixture changed")
    if not args.smoke:
        require(digest(snapshot_path / "model.safetensors") == WEIGHTS_SHA256, "checkpoint pin differs")
    summary["fixtures"] = manifest
    worker.configs(evidence.directory / "models", args.source, args.fixtures, "B", snapshot_path, smoke=args.smoke)
    app = worker.AppFactory.create_app(
        worker.AppStateConfig(
            models_dir=evidence.directory / "models",
            device="cpu" if args.smoke else "cuda:0",
            remote_serving=False,
            bind_host="127.0.0.1",
        )
    )
    sock = socket.socket()
    task = None
    client = None
    server = None
    adapter = None
    try:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=20, access_log=False)
        )
        task = asyncio.create_task(server.serve(sockets=[sock]))
        client = worker.Client(url)
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("server exited before readiness")
            await asyncio.sleep(0.05)
        registry = app.state.registry
        adapter = await registry.load_async(MODEL, "cpu" if args.smoke else "cuda:0")
        model_worker = registry._loaded[MODEL].worker
        require(
            model_worker is not None and getattr(adapter, "_lora", None) is None, "baseline worker/manager mismatch"
        )
        state = await asyncio.to_thread(worker.tensor_state, adapter)
        expected_dtype = "torch.float32" if args.smoke else "torch.float16"
        require({x["dtype"] for x in state["tensors"].values()} == {expected_dtype}, "baseline precision differs")
        summary["actual_precision"] = expected_dtype
        summary["original_tensor_state"] = state
        summary["loaded_before_forward_memory"] = worker.memory()
        bases = [await asyncio.to_thread(client.score, None, f"base-{i}") for i in range(3)]
        baseline = bases[0]["scores"]
        require(max(vector_delta(baseline, row["scores"]) for row in bases) <= RESTORE_ATOL, "unstable base scores")
        if not args.smoke:
            require(vector_delta(baseline, CPU_ORACLE) <= ORACLE_ATOL, "finite checkpoint oracle differs")
        summary["base_scores"] = baseline
        summary["shape_oracles"] = await worker.oracle(adapter, model_worker, None, smoke=args.smoke)
        summary["initial_memory"] = worker.memory()
        summary["untimed_replays"] = {}
        adapter.observation_phase = "untimed-stability-setup"
        for concurrency in CONCURRENCIES:
            first = len(adapter.batch_observations)
            rows = await asyncio.to_thread(worker.replay, url, concurrency)
            batches = adapter.batch_observations[first:]
            require(bool(batches), "missing actual untimed batch observations")
            worker.validate_usage(rows, batches)
            for batch in batches:
                require_forward_selection(batch, None)
                require(
                    vector_delta(batch["scores"], summary["shape_oracles"][worker.batch_key(batch)]["scores"])
                    <= RESTORE_ATOL,
                    "untimed shape oracle differs",
                )
            summary["untimed_replays"][str(concurrency)] = {"requests": rows, "batches": batches}
        # These three setup memory observations mirror the shared B harness.
        # There is no new forced collection between any diagnostic windows.
        summary["resident_memory"] = worker.memory()
        summary["windows"] = []
        continuity = dict(summary["required_telemetry_preflight"])
        with worker.unobserved(adapter):
            for concurrency in CONCURRENCIES:
                for window, warmups in WINDOWS:
                    directory = evidence.directory / f"C{concurrency}" / window
                    row = await asyncio.to_thread(
                        measure_window,
                        url,
                        concurrency,
                        window,
                        warmups,
                        directory,
                        baseline,
                        summary["shape_oracles"],
                        smoke=args.smoke,
                        continuity=continuity,
                    )
                    summary["windows"].append(row)
                    atomic_json(evidence.directory / "process.json", summary)
                    # Only compact metadata survives here. Raw requests live on
                    # disk; ordinary reference release and automatic GC apply.
        summary["restored_tensor_state"] = await asyncio.to_thread(worker.tensor_state, adapter)
        worker.base_unchanged(state, summary["restored_tensor_state"], restored=True)
        await registry.unload_async(MODEL)
        summary["completed"] = True
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            try:
                if server is not None and task is not None:
                    server.should_exit = True
                    await asyncio.wait_for(asyncio.shield(task), timeout=30)
                    summary["shutdown_complete"] = True
            finally:
                sock.close()


async def child(args: argparse.Namespace) -> dict[str, Any]:
    evidence = DiagnosticEvidence(args.output.resolve())
    summary: dict[str, Any] = {
        "qualification": False,
        "diagnostic_only": True,
        "completed": False,
        "shutdown_complete": False,
        "process": args.process,
        "pid": os.getpid(),
        "source_commit": BASE_REVISION,
        "smoke": args.smoke,
    }
    os.environ.update(
        {
            "SIE_TELEMETRY_DISABLED": "1",
            "SIE_DISK_CACHE_ENABLED": "false",
            "SIE_MAX_LORAS_PER_MODEL": "2",
            "SIE_MAX_BATCH_REQUESTS": str(MAX_BATCH_ITEMS),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    try:
        summary["required_telemetry_preflight"] = snapshot()
        environment = worker.preflight(args, evidence)
        environment["hardware_preflight_only"] = environment.pop("qualification")
        environment["qualification"] = False
        environment["source_imports"] = {}
        for name, target in (("sdk", worker.SIEClient), ("server", worker.AppFactory)):
            module_file = sys.modules[target.__module__].__file__
            assert module_file is not None, "source module has no file"
            source_file = Path(module_file).resolve()  # noqa: ASYNC240 -- preflight before model/server starts
            require(source_file.is_relative_to(args.source.resolve()), "SDK/server imported outside baseline archive")
            environment["source_imports"][name] = {
                "relative_path": str(source_file.relative_to(args.source.resolve())),
                "sha256": digest(source_file),
            }
        environment["source_imports"]["adapter"] = {
            "relative_path": str(Path(environment["adapter_source"]).relative_to(args.source.resolve())),
            "sha256": environment["adapter_sha256"],
        }
        summary["environment"] = environment
        async with asyncio.timeout(args.trial_deadline_s):
            await serve_windows(args, evidence, summary)
    except BaseException as error:
        summary["completed"] = False
        summary["error"] = worker.error_record(error)
        evidence.emit("failure", **summary["error"])
        raise
    finally:
        atomic_json(evidence.directory / "process.json", summary)
        evidence.emit("finished", completed=summary["completed"], shutdown_complete=summary["shutdown_complete"])
    return summary


def reopen_window(directory: Path, record: dict[str, Any]) -> dict[str, Any]:
    relative = f"C{record['concurrency']}/{record['window']}/window.json"
    require(record["record"] == relative, "window path differs from identity")
    path = directory / relative
    require(
        path.resolve().is_relative_to(directory.resolve()) and digest(path) == record["sha256"],
        "window path/hash differs",
    )
    window = json.loads(path.read_text())
    require(window["qualification"] is False and window["completed"] is True, "window is incomplete or qualified")
    require(
        (window["window"], window["concurrency"]) == (record["window"], record["concurrency"]),
        "window identity differs",
    )
    require(
        window["warmups"] == (2 if window["smoke"] else dict(WINDOWS)[window["window"]]), "fixed warmup count differs"
    )
    attempt = path.parent / "events.jsonl"
    require(digest(attempt) == window["attempt_sha256"], "raw attempt hash differs")
    verify_attempt(attempt, window)
    recalculated = metrics(window["performance"])
    require(recalculated == window["metrics"] == record["metrics"], "window compact projection differs")
    require(
        telemetry_delta(window["telemetry_before"], window["telemetry_after"]) == window["telemetry_delta"],
        "telemetry projection mismatch",
    )
    require(
        chronological_quartiles(window["performance"]["samples"]) == window["chronological_quartiles"],
        "quartile projection mismatch",
    )
    return window


def verify_process_sources(rows: list[dict[str, Any]], source: Path) -> None:
    require(bool(rows), "missing process environments")
    first = rows[0]
    fields = (
        "versions",
        "torchao",
        "python",
        "threads",
        "device",
        "torch_cuda",
        "installed_distributions",
        "installed_distributions_sha256",
        "source_imports",
        "hardware_preflight_only",
    )
    for row in rows:
        require(row["source_commit"] == BASE_REVISION and row["smoke"] == first["smoke"], "source/mode changed")
        environment = row["environment"]
        require(environment["qualification"] is False, "environment incorrectly marks qualification")
        require(
            environment["installed_distributions_sha256"] == fingerprint(environment["installed_distributions"]),
            "installed distribution manifest hash differs",
        )
        require(all(environment[key] == first["environment"][key] for key in fields), "process environments differ")
        require(set(environment["source_imports"]) == {"adapter", "server", "sdk"}, "missing source import proof")
        for imported in environment["source_imports"].values():
            path = source / imported["relative_path"]
            require(
                path.resolve().is_relative_to(source.resolve()) and digest(path) == imported["sha256"],
                "imported source differs from immutable baseline",
            )
        require(
            row["actual_precision"] == first["actual_precision"] and row["fixtures"] == first["fixtures"],
            "process model or precision differs",
        )
        initial, reference = row["required_telemetry_preflight"], first["required_telemetry_preflight"]
        require(
            initial["core"]["gc_thresholds"] == reference["core"]["gc_thresholds"]
            and initial["affinity"] == reference["affinity"],
            "cross-process GC configuration/affinity differs",
        )
        require(
            all(initial["cgroup"].get(key) == reference["cgroup"].get(key) for key in ("path", "device", "inode"))
            and all(initial["cgroup_files"].get(key) == reference["cgroup_files"].get(key) for key in CONTROL_FILES),
            "cross-process cgroup identity/configuration differs",
        )
        require(
            vector_delta(row["base_scores"], first["base_scores"]) <= RESTORE_ATOL, "cross-process base scores differ"
        )
        require(set(row["shape_oracles"]) == set(first["shape_oracles"]), "cross-process oracle workloads differ")
        for key, value in row["shape_oracles"].items():
            require(
                vector_delta(value["scores"], first["shape_oracles"][key]["scores"]) <= RESTORE_ATOL,
                "cross-process shape oracle differs",
            )


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(
        len(rows) == PROCESSES and [row["process"] for row in rows] == list(range(PROCESSES)),
        "six process units required",
    )
    require(len({row["pid"] for row in rows}) == PROCESSES, "fresh process identities required")
    require(
        all(row["qualification"] is False and row["completed"] and row["shutdown_complete"] for row in rows),
        "incomplete diagnostic process",
    )
    schedule = [(c, w) for c in CONCURRENCIES for w, _ in WINDOWS]
    require(
        all([(w["concurrency"], w["window"]) for w in row["windows"]] == schedule for row in rows),
        "all54 fixed windows required",
    )
    gates, dispersion = {}, {}
    for concurrency, field in METRICS:
        name = f"C{concurrency}_{field}"
        values = {
            window: [
                next(w for w in row["windows"] if w["concurrency"] == concurrency and w["window"] == window)["metrics"][
                    field
                ]
                for row in rows
            ]
            for window, _ in WINDOWS
        }
        require(
            all(math.isfinite(x) and x > 0 for series in values.values() for x in series), "invalid statistical operand"
        )
        ratio = [a / b if field == "rps" else b / a for a, b in zip(values["L1"], values["L2"], strict=True)]
        gates[name] = {
            "L2_over_L1_degradation": paired_gate(ratio),
            "L1_over_L2_degradation": paired_gate([1 / value for value in ratio]),
        }
        dispersion[name] = {}
        for window, series in values.items():
            dispersion[name][window] = {
                "values_by_process": series,
                "min": min(series),
                "median": statistics.median(series),
                "max": max(series),
                "max_over_min": max(series) / min(series),
                "arithmetic_cv": statistics.stdev(series) / statistics.mean(series),
                "log_sd": statistics.stdev([math.log(x) for x in series]),
            }
    passed = all(gate["passed"] for metric in gates.values() for gate in metric.values())
    return {
        "qualification": False,
        "statistical_stability_passed": passed,
        "directional_gates": gates,
        "between_process_dispersion": dispersion,
        "resource_telemetry_complete": all(w["resource_telemetry_complete"] for row in rows for w in row["windows"]),
        "interpretation": "Symmetric later-window agreement only; no causal warmup claim or fresh-process candidate qualification",
    }


def admission(now: float, deadline: float, trial_limit: float) -> dict[str, float]:
    reserve = trial_limit + WAIT_GRACE_SECONDS + TERMINATE_GRACE_SECONDS + KILL_GRACE_SECONDS
    require(now + reserve < deadline, "insufficient time for complete diagnostic worker and cleanup")
    return {
        "monotonic_s": now,
        "deadline_monotonic_s": deadline,
        "reserve_s": reserve,
        "margin_s": deadline - now - reserve,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = extract(args.inputs.resolve(), args.output / "sources")
    summary: dict[str, Any] = {
        "schema": 1,
        "qualification": False,
        "diagnostic_only": True,
        "completed": False,
        "statistical_stability_passed": False,
        "smoke": args.smoke,
        "source_manifest": manifest,
        "executed_source": "B",
        "harness_manifest": harness_manifest(),
        "processes": [],
    }
    if args.environment_receipt is not None:
        summary["environment_preparation"] = {
            "sha256": digest(args.environment_receipt),
            "receipt": json.loads(args.environment_receipt.read_text()),
        }
    deadline = started + args.deadline_s
    summary["runner_clock"] = {
        "monotonic_start_s": started,
        "wall_start_ns": time.time_ns(),
        "deadline_monotonic_s": deadline,
        "budget_s": args.deadline_s,
    }
    summary["methods"] = {
        "window_order": [{"name": name, "sequential_warmup_calls": count} for name, count in WINDOWS],
        "phase_order": list(CONCURRENCIES),
        "fresh_baseline_processes": PROCESSES,
        "gc": "Original automatic GC; no extra per-window collections, disabling or callbacks",
        "raw_retention": "Per-window raw disk records; previous payload references are released normally",
        "telemetry_scope": "Whole performance call including warmup, drain, validation and raw emission; not timed-only",
        "qualification": False,
    }
    try:
        for index in range(1 if args.smoke else PROCESSES):
            reserved = admission(time.monotonic(), deadline, args.trial_deadline_s)
            directory = args.output / f"process-{index}"
            source = (args.output / "sources/B").resolve()
            command = [
                sys.executable,
                "-m",
                "tools.benchmarks.cross_encoder_lora.stability",
                "--worker",
                "--source",
                str(source),
                "--process",
                str(index),
                "--fixtures",
                str(args.fixtures.resolve()),
                "--output",
                str(directory.resolve()),
                "--trial-deadline-s",
                str(args.trial_deadline_s),
            ]
            if args.smoke:
                command.append("--smoke")
            print(f"Starting baseline process {index + 1}", flush=True)
            process = None
            error = None
            with (args.output / f"process-{index}.log").open("w") as log:
                try:
                    process = subprocess.Popen(
                        command, env=source_environment(source), cwd=source, stdout=log, stderr=subprocess.STDOUT
                    )
                    process.wait(timeout=args.trial_deadline_s + WAIT_GRACE_SECONDS)
                except BaseException as caught:  # noqa: BLE001 -- persist error after unconditional cleanup, then re-raise
                    error = caught
                finally:
                    if process is not None:
                        stop(process)
            terminal = directory / "process.json"
            record = (
                json.loads(terminal.read_text())
                if terminal.exists()
                else {"qualification": False, "completed": False, "shutdown_complete": False}
            )
            row = {
                "qualification": False,
                "process": index,
                "pid": record.get("pid"),
                "completed": record["completed"] and error is None,
                "shutdown_complete": record["shutdown_complete"],
                "returncode": process.returncode if process is not None else None,
                "admission": reserved,
                "record": f"process-{index}/process.json",
                "sha256": digest(terminal) if terminal.exists() else None,
            }
            if error is not None:
                row["error"] = worker.error_record(error)
            summary["processes"].append(row)
            atomic_json(args.output / "summary.json", summary)
            if error is not None:
                raise error
            require(
                row["returncode"] == 0 and row["completed"] and row["shutdown_complete"],
                "baseline process failed; evidence retained",
            )
            print(f"Finished baseline process {index + 1}", flush=True)
        complete = []
        for row in summary["processes"]:
            path = args.output / row["record"]
            require(digest(path) == row["sha256"], "process record changed")
            record = json.loads(path.read_text())
            require(record["process"] == row["process"] and record["pid"] == row["pid"], "process identity mismatch")
            previous = record["required_telemetry_preflight"]
            for entry in record["windows"]:
                reopened = reopen_window(path.parent, entry)
                telemetry_delta(previous, reopened["telemetry_before"])
                previous = reopened["telemetry_after"]
                entry["metrics"] = metrics(reopened["performance"])
                require(
                    entry["resource_telemetry_complete"] == reopened["telemetry_delta"]["resource_telemetry_complete"],
                    "telemetry projection mismatch",
                )
            complete.append(record)
        verify_process_sources(complete, (args.output / "sources/B").resolve())
        summary["completed"] = True
        if not args.smoke:
            summary["analysis"] = analyze(complete)
            summary["statistical_stability_passed"] = summary["analysis"]["statistical_stability_passed"]
            require(summary["statistical_stability_passed"], "fixed symmetric baseline stability gates did not pass")
    except BaseException as error:
        summary["error"] = worker.error_record(error)
        raise
    finally:
        atomic_json(args.output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-s", type=float, default=RUNNER_LIMIT)
    parser.add_argument("--trial-deadline-s", type=float, default=WORKER_LIMIT)
    parser.add_argument("--environment-receipt", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--process", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(
        0 < args.trial_deadline_s <= WORKER_LIMIT and 0 < args.deadline_s <= RUNNER_LIMIT,
        "diagnostic deadline outside fixed bounds",
    )
    if args.worker:
        require(args.source is not None, "worker requires exact source")
        asyncio.run(child(args))
    else:
        require(args.inputs is not None, "runner requires immutable inputs")
        signal.signal(signal.SIGTERM, terminate)
        run(args)


if __name__ == "__main__":
    main()
