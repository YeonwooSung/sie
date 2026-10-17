"""Real SDK-only child protocol, identity, clock and owned-process controls."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
from sie_sdk._msgpack import packb, unpackb

from tools.benchmarks.cross_encoder_lora.evidence import Evidence, atomic_json, digest
from tools.benchmarks.cross_encoder_lora.isolation import IsolatedLoadgen
from tools.benchmarks.cross_encoder_lora.loadgen import PHASES
from tools.benchmarks.cross_encoder_lora.protocol import MODEL
from tools.benchmarks.cross_encoder_lora.runner import ROOT, source_environment


@pytest.fixture
def sdk_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            body = unpackb(self.rfile.read(int(self.headers["content-length"])), numeric_arrays=False)
            content = packb(
                {
                    "model": MODEL,
                    "scores": [
                        {"item_id": item["id"], "score": float(4 - index), "rank": index}
                        for index, item in enumerate(body["items"])
                    ],
                    "usage": {"input_tokens": 17},
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/msgpack")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Total-Time", "0.01")
            self.end_headers()
            self.wfile.write(content)
            self.wfile.flush()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    monkeypatch.setenv("PYTHONPATH", source_environment(ROOT)["PYTHONPATH"])
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_one_sdk_only_exec_pid_reused_across_all_phases_and_reaped(sdk_server: str, tmp_path: Path) -> None:
    evidence = Evidence(tmp_path / "out")
    async with IsolatedLoadgen(sdk_server, ROOT, evidence, smoke=True) as owner:
        assert owner.process is not None
        pid = owner.process.pid
        assert pid != os.getpid()
        for concurrency in PHASES:
            await owner.phase(concurrency)
            assert owner.process.pid == pid
        with pytest.raises(AssertionError, match="reaped"):
            owner.results()
    assert owner.receipt["passed"]
    assert owner.receipt["reaped"]
    assert owner.receipt["returncode"] == 0
    identity = owner.receipt["identity"]
    assert identity["forbidden_imports_present"] is False
    assert identity["process_group"] == os.getpgrp()
    assert identity["ppid"] == os.getpid()
    assert Path(identity["sdk_path"]).is_relative_to(ROOT / "packages/sie_sdk/src")
    results = owner.results()
    assert list(results) == ["1", "4", "16"]
    assert results["1"]["count"] == 4
    for phase in results.values():
        assert phase["identity"] == identity
        assert phase["client_observation"]["gc"]["integrity_passed"]
        assert phase["server_observation"]["gc"]["integrity_passed"]
        assert phase["measurement_clock"] == phase["server_observation"]["clock"]
        assert phase["observer_cost_gate_applied"] is False  # CPU smoke never qualifies performance.
        assert all(len(row["responses"]) == 1 for row in phase["samples"])
    terminal = json.loads((evidence.directory / "loadgen/terminal.json").read_text())
    assert terminal["phases"] == owner.records
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def synthetic_owner(tmp_path: Path) -> tuple[IsolatedLoadgen, dict[str, Any], dict[str, Any]]:
    owner = IsolatedLoadgen("http://unused.invalid", ROOT, Evidence(tmp_path / "out"), smoke=False)
    identity = {"pid": 999999, "source": str(ROOT)}
    owner.receipt["identity"] = identity
    clock = {"start": 1.0, "deadline": 61.0, "end": 61.1}

    def observation(pid: int) -> dict[str, Any]:
        return {
            "before": {"core": {"pid": pid}},
            "after": {"core": {"pid": pid}},
            "clock": clock,
            "gc": {
                "pid": pid,
                "integrity_passed": True,
                "window_start_ns": 1_000_000_000,
                "window_end_ns": 61_100_000_000,
                "estimated_observer_fraction": 0.0001,
            },
        }

    owner.observations[1] = observation(os.getpid())
    record = {"concurrency": 1}
    value = {
        "concurrency": 1,
        "smoke": False,
        "identity": identity,
        "measurement_clock": clock,
        "client_observation": observation(identity["pid"]),
    }
    return owner, record, value


@pytest.mark.parametrize(
    "mutation",
    [
        "swapped_phase",
        "clock",
        "observation_clock",
        "identity",
        "pid",
        "window",
        "cost_nan",
        "combined_budget",
        "integrity",
    ],
)
def test_phase_payload_must_match_observed_ipc_provenance_and_combined_budget(tmp_path: Path, mutation: str) -> None:
    owner, record, original = synthetic_owner(tmp_path)
    value = copy.deepcopy(original)
    assert owner._validate_phase(record, copy.deepcopy(original))["observer_cost_gate_applied"]
    if mutation == "swapped_phase":
        value["concurrency"] = 4
    elif mutation == "clock":
        value["measurement_clock"]["end"] += 1
    elif mutation == "observation_clock":
        value["client_observation"]["clock"] = {"start": 2.0, "deadline": 62.0, "end": 62.1}
    elif mutation == "identity":
        value["identity"]["source"] = "/wrong"
    elif mutation == "pid":
        value["client_observation"]["after"]["core"]["pid"] += 1
    elif mutation == "window":
        value["client_observation"]["gc"]["window_start_ns"] += 1
    elif mutation == "cost_nan":
        value["client_observation"]["gc"]["estimated_observer_fraction"] = float("nan")
    elif mutation == "combined_budget":
        owner.observations[1]["gc"]["estimated_observer_fraction"] = 0.0006
        value["client_observation"]["gc"]["estimated_observer_fraction"] = 0.0006
    else:
        value["client_observation"]["gc"]["integrity_passed"] = False
    with pytest.raises(AssertionError):
        owner._validate_phase(record, value)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["failed", "identity", "schedule", "smoke"])
async def test_valid_terminal_hash_cannot_mask_wrong_terminal_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    owner, _, _ = synthetic_owner(tmp_path)
    owner.records.extend({"concurrency": c} for c in PHASES)
    terminal: dict[str, Any] = {
        "passed": True,
        "identity": owner.receipt["identity"],
        "phases": owner.records,
        "smoke": False,
        "finished_ns": 1,
    }
    terminal = copy.deepcopy(terminal)
    if mutation == "failed":
        terminal["passed"] = False
    elif mutation == "identity":
        terminal["identity"]["pid"] += 1
    elif mutation == "schedule":
        terminal["phases"].reverse()
    else:
        terminal["smoke"] = True
    path = owner.evidence.directory / "loadgen/terminal.json"
    path.parent.mkdir()
    atomic_json(path, terminal)

    class Exited:
        returncode = 0
        stdin = None

        async def wait(self) -> int:
            return 0

    async def send(*_args: Any) -> None:
        pass

    async def read(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"terminal_sha256": digest(path)}

    owner.process = cast("asyncio.subprocess.Process", Exited())
    monkeypatch.setattr(owner, "_send", send)
    monkeypatch.setattr(owner, "_read", read)
    with pytest.raises(AssertionError, match="terminal"):
        await owner._cleanup(True)
    assert owner.receipt["passed"] is False
    assert owner.receipt["reaped"] is True


@pytest.mark.asyncio
async def test_cancel_during_finish_handshake_reaps_silent_real_child(tmp_path: Path) -> None:
    owner = IsolatedLoadgen("http://unused.invalid", ROOT, Evidence(tmp_path / "out"), smoke=True)
    owner.process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        close_fds=True,
    )
    owner.records.extend({"concurrency": c} for c in PHASES)
    task = asyncio.create_task(owner.__aexit__(None, None, None))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=8)
    assert owner.receipt["reaped"]
    assert owner.process is not None
    assert owner.process.returncode is not None


@pytest.mark.asyncio
async def test_child_crash_and_parent_phase_cancellation_reap(sdk_server: str, tmp_path: Path) -> None:
    owner = IsolatedLoadgen(sdk_server, ROOT, Evidence(tmp_path / "out"), smoke=True)
    with pytest.raises(AssertionError, match="missing"):
        async with owner:
            assert owner.process is not None
            owner.process.kill()
            await owner.process.wait()
            await owner._read("ready", 1)
    assert owner.receipt["reaped"]
    assert owner.process is not None
    assert owner.process.returncode is not None


@pytest.mark.asyncio
async def test_termination_exit_race_preserves_original_failure(tmp_path: Path) -> None:
    owner = IsolatedLoadgen("http://unused.invalid", ROOT, Evidence(tmp_path / "out"), smoke=True)

    class RacedExit:
        returncode = None
        stdin = None

        def terminate(self) -> None:
            raise ProcessLookupError("already exited")

        async def wait(self) -> int:
            self.returncode = 1
            return 1

    owner.process = cast("asyncio.subprocess.Process", RacedExit())
    original = ValueError("primary phase failure")
    await owner.__aexit__(ValueError, original, None)
    assert owner.receipt["termination_raced_exit"]
    assert owner.receipt["reaped"]
    assert owner.receipt["returncode"] == 1


@pytest.mark.asyncio
async def test_cleanup_write_failure_never_masks_primary_phase_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.benchmarks.cross_encoder_lora import isolation

    owner = IsolatedLoadgen("http://unused.invalid", ROOT, Evidence(tmp_path / "out"), smoke=True)

    def fail(*_args: Any) -> None:
        raise OSError("injected receipt write failure")

    monkeypatch.setattr(isolation, "atomic_json", fail)
    original = ValueError("original phase error")
    await owner.__aexit__(ValueError, original, None)
    assert any("receipt write failure" in note for note in original.__notes__)
    assert owner.receipt["reaped"]


def test_sdk_calls_lanes_and_measured_executor_core_match_recorded_harness() -> None:
    from tools.benchmarks.cross_encoder_lora import client

    baseline = subprocess.run(
        ["git", "show", "a2932d81:tools/benchmarks/cross_encoder_lora/worker.py"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if baseline.returncode:
        pytest.skip("source invariant requires the recorded parent commit in git history")
    old, new = ast.parse(baseline.stdout), ast.parse(Path(client.__file__).read_text())

    def definition(tree: ast.AST, name: str) -> ast.AST:
        return next(
            node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
        )

    class Names(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            names = {
                "sequential": "SEQUENTIAL",
                "steady_seconds": "STEADY_SECONDS",
                "future_timeout": "PERFORMANCE_FUTURE_TIMEOUT_S",
            }
            return ast.Name(id=names.get(node.id, node.id), ctx=node.ctx)

    assert ast.dump(definition(old, "Client")) == ast.dump(definition(new, "Client"))
    assert ast.dump(definition(old, "lane")) == ast.dump(Names().visit(definition(new, "lane")))

    def executor(tree: ast.AST) -> ast.With:
        return next(
            node
            for node in ast.walk(definition(tree, "performance"))
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "id", None) == "ThreadPoolExecutor"
                for item in node.items
            )
        )

    new_executor = executor(new)
    # Only the pre-clock readiness hook is new. It aborts the barrier on failure.
    assert isinstance(new_executor.body[1], ast.Try)
    new_executor.body.pop(1)
    assert ast.dump(executor(old)) == ast.dump(Names().visit(new_executor))
