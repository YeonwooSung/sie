"""Failure-path checks for recovery evidence and owned server shutdown."""

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from tools.benchmarks.model_recovery import probe


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["enumerate", "identity"])
async def test_process_inspection_failure_does_not_skip_shutdown_or_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    monkeypatch.setattr(probe.os, "environ", probe.os.environ.copy())
    registry = SimpleNamespace(_loaded={}, loaded_model_names=[])
    app = SimpleNamespace(state=SimpleNamespace(registry=registry))
    monkeypatch.setattr(probe.AppFactory, "create_app", lambda _config: app)
    monkeypatch.setattr(probe, "snapshot", lambda _registry: {"models": {}})

    class Server:
        started = True
        should_exit = False
        shutdown_complete = False

        async def serve(self, **_kwargs: object) -> None:
            await asyncio.sleep(0)
            assert self.should_exit
            self.shutdown_complete = True

    server = Server()
    monkeypatch.setattr(probe.uvicorn, "Server", lambda _config: server)

    class Child:
        pid = 12345

        def create_time(self) -> float:
            raise psutil.AccessDenied(self.pid)

    class Parent:
        def children(self, **_kwargs: object) -> list[Child]:
            if failure_stage == "enumerate":
                raise psutil.AccessDenied
            return [Child()]

    monkeypatch.setattr(probe.psutil, "Process", lambda _pid: Parent())
    args = argparse.Namespace(smoke=True, deadline_s=10, output=tmp_path / "evidence")
    evidence = probe.Evidence(args.output)

    # The empty registry forces an ordinary protocol failure before warmup.
    # Process inspection must not replace that failure or bypass teardown.
    assert await probe.run(args, evidence) is False
    assert server.should_exit
    assert server.shutdown_complete
    assert any(row["event"] == "protocol_failed" for row in evidence.rows)
    inspection = next(row for row in evidence.rows if row["event"] == "before_shutdown")
    assert inspection["inspection_errors"][0]["stage"] == failure_stage
    assert inspection["inspection_errors"][0]["error_type"] == "AccessDenied"
    cleanup = next(row for row in evidence.rows if row["event"] == "cleanup")
    assert cleanup["okay"] is False
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["passed"] is False
    assert summary["cleanup_ok"] is False
