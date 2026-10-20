"""The serve CLI preserves cache policy unless an explicit flag overrides it."""

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from sie_sdk.cache import get_cache_config
from sie_server import cli
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    saved = dict(os.environ)
    for name in tuple(os.environ):
        if name.startswith("SIE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path / "data"))
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize("environment", [None, "true", "false", "1", "0", "yes", "no"])
@pytest.mark.parametrize("flag", [None, "--hf-fallback", "--no-hf-fallback"])
def test_cli_cache_policy_precedence(
    monkeypatch: pytest.MonkeyPatch, environment: str | None, flag: str | None
) -> None:
    if environment is not None:
        monkeypatch.setenv("SIE_HF_FALLBACK", environment)
    observed = []
    run_server = Mock(side_effect=lambda **_kwargs: observed.append(get_cache_config().hf_fallback))
    monkeypatch.setattr(cli, "run_server", run_server)
    args = ["serve", "--host", "127.0.0.1"]
    if flag is not None:
        args.append(flag)

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    expected = (
        flag == "--hf-fallback" if flag is not None else environment is None or environment in {"true", "1", "yes"}
    )
    assert observed == [expected]
    assert os.environ["SIE_HF_FALLBACK"] == str(expected).lower()
    run_server.assert_called_once()
