"""Admission reserves the complete worker wait and both cleanup grace periods."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.benchmarks.cross_encoder_lora import runner


def arguments(tmp_path: Path, budget: float) -> argparse.Namespace:
    return argparse.Namespace(
        output=tmp_path / "evidence",
        inputs=tmp_path / "inputs",
        fixtures=tmp_path / "fixtures",
        snapshot=None,
        smoke=True,
        environment_receipt=None,
        deadline_s=budget,
        trial_deadline_s=120,
    )


def setup_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "extract", lambda *_: {"sources": {"candidate": {"commit": "frozen"}}})
    monkeypatch.setattr(runner, "harness_manifest", dict)


@pytest.mark.parametrize(("budget", "admitted"), [(209.999, False), (210.0, False), (210.001, True)])
def test_trial_admission_reserves_full_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: float,
    admitted: bool,
) -> None:
    setup_sources(monkeypatch)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 100.0)
    waits = []
    started = []

    class CompletedProcess:
        returncode = 0

        def wait(self, *, timeout: float) -> int:
            waits.append(timeout)
            return self.returncode

        def poll(self) -> int:
            return self.returncode

    def start(command: list[str], **_kwargs: Any) -> CompletedProcess:
        started.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "trial.json").write_text(
            json.dumps({"pair": 0, "arm": "R", "passed": True, "shutdown_complete": True})
        )
        return CompletedProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    args = arguments(tmp_path, budget)
    if admitted:
        assert runner.run(args)["passed"]
        assert waits == [165]
    else:
        with pytest.raises(AssertionError, match="insufficient time"):
            runner.run(args)
        terminal = json.loads((args.output / "summary.json").read_text())
        assert not terminal["passed"]
        assert terminal["trials"] == []
    assert bool(started) is admitted


def test_full_timeout_cleanup_finishes_before_budget_and_retains_partial_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_sources(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    actions = []

    class HungProcess:
        returncode = None

        def wait(self, *, timeout: float) -> int:
            actions.append(("wait", timeout))
            clock[0] += timeout
            if len([action for action, _ in actions if action == "wait"]) < 3:
                raise subprocess.TimeoutExpired("owned-worker", timeout)
            self.returncode = -9
            return self.returncode

        def poll(self) -> None:
            return None

        def send_signal(self, signum: int) -> None:
            actions.append(("signal", signum))

        def kill(self) -> None:
            actions.append(("kill", None))

    def start(command: list[str], **_kwargs: Any) -> HungProcess:
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        (output / "trial.json").write_text(
            json.dumps(
                {
                    "pair": 0,
                    "arm": "R",
                    "passed": False,
                    "shutdown_complete": False,
                    "performance": {"1": {"count": 7}},
                }
            )
        )
        return HungProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    args = arguments(tmp_path, 211)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run(args)
    assert actions == [("wait", 165), ("signal", signal.SIGTERM), ("wait", 35), ("kill", None), ("wait", 10)]
    assert clock[0] == 310
    assert clock[0] < 100 + args.deadline_s
    terminal = json.loads((args.output / "summary.json").read_text())
    assert not terminal["passed"]
    trial = terminal["trials"][0]
    assert trial["performance"]["1"]["count"] == 7
    assert trial["process_returncode"] == -9
    assert trial["runner_error"]["type"] == "TimeoutExpired"
