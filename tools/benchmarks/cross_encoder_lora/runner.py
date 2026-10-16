"""Run the frozen six-block experiment in eighteen owned fresh processes."""

# Only the current Python executable and this checkout's benchmark module run.
# ruff: noqa: S603

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest, fingerprint, vector_delta
from tools.benchmarks.cross_encoder_lora.protocol import (
    BASE_REVISION,
    MAX_EXPERIMENT_SECONDS,
    MAX_TRIAL_SECONDS,
    ORDERS,
    RESTORE_ATOL,
    compare,
    require,
)

ROOT = Path(__file__).resolve().parents[3]


def harness_manifest() -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    files = {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".md"}
        and not any(part in {"_inputs", "_evidence", "__pycache__"} for part in path.relative_to(directory).parts)
    }
    return {"files": files, "file_manifest_sha256": fingerprint(files)}


def source_environment(source: Path) -> dict[str, str]:
    # The harness stays fixed; both server and SDK imports come from the
    # selected immutable archive. Do not run uv sync per arm.
    paths = [source / f"packages/{package}/src" for package in ("sie_server", "sie_sdk", "sie_config", "sie_bundles")]
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(path) for path in [*paths, ROOT]),
        "TOKENIZERS_PARALLELISM": "false",
        "SIE_BUNDLE": "default",
    }


def extract(inputs: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((inputs / "manifest.json").read_text())
    require(manifest["schema"] == 1 and manifest["dependency_lock_identical"], "unexpected source manifest")
    require(manifest["sources"]["B"]["commit"] == BASE_REVISION, "baseline differs from protocol")
    for name, record in manifest["sources"].items():
        require(name in {"B", "candidate"}, "unknown source arm")
        path = inputs / record["archive"]
        require(path.resolve().is_relative_to(inputs.resolve()), "archive path escapes inputs")
        require(digest(path) == record["sha256"], "source archive hash differs")
        target = output / name
        target.mkdir(parents=True)
        with tarfile.open(path) as archive:
            archive.extractall(target, filter="data")
    require(
        (output / "B/uv.lock").read_bytes() == (output / "candidate/uv.lock").read_bytes(), "extracted locks differ"
    )
    return manifest


def cross_checks(trials: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for pair in range(len(ORDERS)):
        rows = {trial["arm"]: trial for trial in trials if trial["pair"] == pair}
        baseline = rows["B"]
        for arm in ("N", "R"):
            candidate = rows[arm]
            require(
                candidate["environment"]["versions"] == baseline["environment"]["versions"],
                "libraries changed between arms",
            )
            require(
                candidate["environment"]["installed_distributions_sha256"]
                == baseline["environment"]["installed_distributions_sha256"],
                "installed distributions changed between arms",
            )
            require(candidate["environment"]["device"] == baseline["environment"]["device"], "GPU changed between arms")
            require(
                vector_delta(candidate["base_scores"], baseline["base_scores"]) <= RESTORE_ATOL,
                "cross-process C1 base scores differ",
            )
            require(
                candidate["initial_memory"]["allocated"] == baseline["initial_memory"]["allocated"],
                "unadapted allocated bytes changed",
            )
            for key, oracle in candidate["shape_oracles"].items():
                require(
                    vector_delta(oracle["scores"], baseline["shape_oracles"][key]["scores"]) <= RESTORE_ATOL,
                    "cross-process shape oracle differs",
                )
            # Actual shape attribution is available only in observed untimed
            # replays. Timed C4/C16 diagnostics are not shape-qualified evidence.
            for phase in candidate["untimed_replays"].values():
                for batch in phase["batches"]:
                    key = fingerprint({"queries": batch["queries"], "documents": batch["documents"]})
                    require(
                        vector_delta(batch["scores"], baseline["shape_oracles"][key]["scores"]) <= RESTORE_ATOL,
                        "candidate untimed SDK replay differs from the matched B shape",
                    )
            delta = candidate["resident_memory"]["allocated"] - baseline["resident_memory"]["allocated"]
            allowance = candidate.get("adapter_storage", {}).get("rounded_storage_bytes", 0)
            require(0 <= delta <= allowance, "resident allocated bytes exceed measured adapter storage allowance")
            results[f"{pair}-{arm}"] = {"allocated_delta": delta, "allowance": allowance, "passed": True}
    return results


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = extract(args.inputs.resolve(), args.output / "sources")
    summary: dict[str, Any] = {
        "schema": 1,
        "passed": False,
        "source_manifest": manifest,
        "harness_manifest": harness_manifest(),
        "smoke": args.smoke,
        "orders": ORDERS,
        "trials": [],
    }
    summary_path = args.output / "summary.json"
    absolute_deadline = time.monotonic() + args.deadline_s
    if args.environment_receipt is not None:
        summary["environment_preparation"] = {
            "sha256": digest(args.environment_receipt),
            "receipt": json.loads(args.environment_receipt.read_text()),
        }
    try:
        schedule = [(0, "R")] if args.smoke else [(pair, arm) for pair, order in enumerate(ORDERS) for arm in order]
        for pair, arm in schedule:
            require(
                time.monotonic() + args.trial_deadline_s + 45 < absolute_deadline,
                "insufficient time for a complete bounded next trial and cleanup",
            )
            directory = args.output / f"trial-{pair}-{arm}"
            source = (args.output / "sources" / ("B" if arm == "B" else "candidate")).resolve()
            command = [
                sys.executable,
                "-m",
                "tools.benchmarks.cross_encoder_lora.worker",
                "--source",
                str(source),
                "--fixtures",
                str(args.fixtures.resolve()),
                "--output",
                str(directory.resolve()),
                "--arm",
                arm,
                "--pair",
                str(pair),
                "--deadline-s",
                str(args.trial_deadline_s),
            ]
            if args.snapshot:
                command.extend(["--snapshot", str(args.snapshot.resolve())])
            if args.smoke:
                command.append("--smoke")
            print(f"Starting block {pair + 1}, arm {arm}", flush=True)
            process = None
            pending_error: BaseException | None = None
            returncode = None
            with (args.output / f"trial-{pair}-{arm}.log").open("w") as log:
                try:
                    process = subprocess.Popen(
                        command, env=source_environment(source), cwd=source, stdout=log, stderr=subprocess.STDOUT
                    )
                    returncode = process.wait(timeout=args.trial_deadline_s + 45)
                except BaseException as error:  # noqa: BLE001 -- persist the trial, then re-raise after owned cleanup
                    pending_error = error
                finally:
                    if process is not None:
                        stop(process)
                        returncode = process.returncode
            trial: dict[str, Any]
            if (directory / "trial.json").is_file():
                trial = json.loads((directory / "trial.json").read_text())
            else:
                trial = {
                    "pair": pair,
                    "arm": arm,
                    "passed": False,
                    "shutdown_complete": False,
                    "error": "worker produced no terminal record",
                }
            trial["process_returncode"] = returncode
            trial["source_commit"] = manifest["sources"]["B" if arm == "B" else "candidate"]["commit"]
            trial["terminal_record_sha256"] = (
                digest(directory / "trial.json") if (directory / "trial.json").exists() else None
            )
            if pending_error is not None:
                trial["passed"] = False
                trial["runner_error"] = {"type": type(pending_error).__name__, "message": str(pending_error)}
            summary["trials"].append(trial)
            atomic_json(summary_path, summary)
            if pending_error is not None:
                raise pending_error
            require(
                returncode == 0 and trial["passed"] and trial["shutdown_complete"],
                "trial failed; raw evidence retained",
            )
            print(f"Finished block {pair + 1}, arm {arm}", flush=True)
        if args.smoke:
            summary["passed"] = True
            summary["qualification"] = False
        else:
            summary["correctness_memory"] = cross_checks(summary["trials"])
            summary["statistics"] = compare(summary["trials"])
            summary["passed"] = summary["statistics"]["passed"]
            require(summary["passed"], "fixed non-inferiority gates did not pass")
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        atomic_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--environment-receipt",
        type=Path,
        help="Optional operator-owned JSON receipt of environment preparation (no secrets)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-s", type=float, default=2400)
    parser.add_argument("--trial-deadline-s", type=float, default=120)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    require(
        0 < args.deadline_s <= MAX_EXPERIMENT_SECONDS and 0 < args.trial_deadline_s <= MAX_TRIAL_SECONDS,
        "deadline outside bounded protocol",
    )
    signal.signal(signal.SIGTERM, terminate)
    run(args)


def terminate(signum: int, _frame: Any) -> None:
    # Enter the ordinary owned-child cleanup and atomic evidence finally paths.
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    main()
