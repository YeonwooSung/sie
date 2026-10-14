"""Fill ``description`` and ``release_date`` into every model config that lacks them.

``/v1/models`` reports both fields for every model, as the System One API's
model list does. This script writes them into ``packages/sie_server/models/*.yaml``
once, so serving never calls Hugging Face:

- ``release_date``: the commit date of the pinned ``hf_revision`` on Hugging
  Face; for a model without one, the date of the repo's latest commit; for a
  model with no Hugging Face repo, the date the config was first committed.
- ``description``: a hand-written line for the typed-decision models below;
  otherwise ``"<hf_id or sie_id>: <task> model."`` from the config's tasks.

The two lines are inserted after ``sie_id:``; the rest of each file is left as
it is. Configs that already set a field keep it. Run from the repository root:

    mise exec -- uv run --frozen python packages/sie_server/scripts/fill_model_metadata.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

DESCRIPTIONS = {
    "fastino/GLiNER2.5-Decide": (
        "English typed-decision classifier (GLiNER2.5, DeBERTa-v3-large): answers noul, choice and score "
        "questions about a text in one pass. 512-token window."
    ),
    "fastino/GLiNER2.5-Decide-1B": (
        "English typed-decision classifier (GLiNER2.5, Ettin 1B encoder): answers noul, choice and score "
        "questions about a text in one pass. 2,048-token window."
    ),
    "fastino/GLiNER2.5-multi-Decide": (
        "Multilingual typed-decision classifier (GLiNER2.5, mDeBERTa-v3-base): answers noul, choice and score "
        "questions about a text in one pass. 2,048-token window."
    ),
    "convaiinnovations/laya": (
        "English typed-decision classifier (Laya): answers noul, choice and score questions, one row per "
        "question. 512-token rows."
    ),
    "convaiinnovations/laya-typed-decisions": (
        "Typed-decision classifier (Laya): answers noul, choice and score questions, one row per question. "
        "1,024-token rows."
    ),
    "convaiinnovations/laya-multilingual": (
        "Multilingual typed-decision classifier (Laya), uncalibrated: answers noul, choice and score "
        "questions, one row per question. 1,024-token rows."
    ),
}

TASK_NAMES = {"encode": "embedding", "score": "reranking", "extract": "extraction", "generate": "generation"}


def describe(config: dict) -> str:
    """The config's description: hand-written, or built from its tasks."""
    sie_id = config["sie_id"]
    if sie_id in DESCRIPTIONS:
        return DESCRIPTIONS[sie_id]
    tasks = [
        TASK_NAMES[task]
        for task, spec in (config.get("tasks") or {}).items()
        if spec is not None and task in TASK_NAMES
    ]
    kind = (", ".join(tasks[:-1]) + " and " + tasks[-1]) if len(tasks) > 1 else (tasks[0] if tasks else "inference")
    return f"{config.get('hf_id') or sie_id}: {kind} model."


def release_date(api: HfApi, config: dict, path: Path) -> str:
    """The release date of the config's weights, YYYY-MM-DD."""
    hf_id = config.get("hf_id")
    if hf_id:
        # The revision's own metadata carries its commit date, and is public for gated repos too.
        info = api.model_info(hf_id, revision=config.get("hf_revision") or "main")
        if info.last_modified is None:
            raise ValueError(f"no commit date for {hf_id}")
        return info.last_modified.date().isoformat()
    first = subprocess.run(
        ["git", "log", "--diff-filter=A", "--follow", "--format=%cs", "--", str(path)],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return first[-1]


def main() -> int:
    api = HfApi()
    failures: list[str] = []
    for path in sorted(MODELS_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        config = yaml.safe_load(text)
        lines: list[str] = []
        if "description" not in config:
            lines.append(f"description: {json.dumps(describe(config), ensure_ascii=False)}")
        if "release_date" not in config:
            try:
                lines.append(f'release_date: "{release_date(api, config, path)}"')
            except Exception as exc:  # noqa: BLE001 — report every model that needs a hand-set date, then fail
                failures.append(f"{path.name}: {exc}")
        if not lines:
            continue
        out: list[str] = []
        inserted = False
        for line in text.splitlines(keepends=True):
            out.append(line)
            if not inserted and line.startswith("sie_id:"):
                out.extend(f"{added}\n" for added in lines)
                inserted = True
        if not inserted:
            failures.append(f"{path.name}: no top-level sie_id line")
            continue
        path.write_text("".join(out), encoding="utf-8")
        print(f"{path.name}: {', '.join(line.split(':', 1)[0] for line in lines)}")
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
