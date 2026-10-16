from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import yaml
from sie_config.model_schema import model_config_schema_errors
from sie_server.config.model import ModelConfig
from sie_server.core.loader import _build_adapter_kwargs, load_adapter

_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "opensearch-project__opensearch-neural-sparse-encoding-doc-v3-gte.yaml"
)
_WEIGHTS_REVISION = "1646fef40807937e8e130c66d327a26421c408d5"
_CODE_REVISION = "e244c9398934a48b4663dacf7ff4965961a6807a"


def _config() -> ModelConfig:
    declaration = yaml.safe_load(_MODEL_PATH.read_text())
    assert model_config_schema_errors(declaration) == []
    return ModelConfig.model_validate(declaration)


def test_catalog_preserves_independent_weights_and_code_revisions() -> None:
    config = _config()
    profile = config.resolve_profile("default")

    assert config.hf_revision == _WEIGHTS_REVISION
    assert profile.loadtime["code_revision"] == _CODE_REVISION
    assert profile.loadtime["trust_remote_code"] is True

    kwargs = _build_adapter_kwargs(config, "float32")
    assert kwargs["revision"] == _WEIGHTS_REVISION
    assert kwargs["code_revision"] == _CODE_REVISION


def test_catalog_code_revision_reaches_tokenizer_and_model(tmp_path: Path) -> None:
    config = _config()
    idf_path = tmp_path / "idf.json"
    idf_path.write_text("{}")
    tokenizer = MagicMock(model_max_length=512, vocab_size=30522, special_tokens_map={}, vocab={})
    model = MagicMock(spec=["new", "config", "to", "eval"])
    model.config = SimpleNamespace(
        vocab_size=30522,
        num_attention_heads=12,
        hidden_size=768,
        max_position_embeddings=512,
    )

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load_tokenizer,
        patch("transformers.AutoModelForMaskedLM.from_pretrained", return_value=model) as load_model,
        patch("huggingface_hub.try_to_load_from_cache", return_value=str(idf_path)),
        patch("huggingface_hub.hf_hub_download") as download,
    ):
        adapter = load_adapter(config, _MODEL_PATH.parent, device="cpu")
        adapter.load("cpu")

    expected = {
        "trust_remote_code": True,
        "revision": _WEIGHTS_REVISION,
        "code_revision": _CODE_REVISION,
    }
    load_tokenizer.assert_called_once_with(config.hf_id, **expected)
    load_model.assert_called_once_with(config.hf_id, torch_dtype=torch.float32, **expected)
    download.assert_not_called()
