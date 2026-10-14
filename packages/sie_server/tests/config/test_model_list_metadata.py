"""Every model config carries the ``description`` and ``release_date`` /v1/models reports.

System One clients (the TypeSafe SDKs) list models and require both fields as
strings. ``packages/sie_server/scripts/fill_model_metadata.py`` fills them into
a new config.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@pytest.mark.parametrize("path", sorted(MODELS_DIR.glob("*.yaml")), ids=lambda path: path.name)
def test_model_config_has_list_metadata(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    fix = "run packages/sie_server/scripts/fill_model_metadata.py"
    assert isinstance(config.get("description"), str), f"no description; {fix}"
    assert config["description"].strip(), f"blank description; {fix}"
    release_date = config.get("release_date")
    assert isinstance(release_date, str), f"release_date must be a quoted YYYY-MM-DD string; {fix}"
    assert dt.date.fromisoformat(release_date).isoformat() == release_date
