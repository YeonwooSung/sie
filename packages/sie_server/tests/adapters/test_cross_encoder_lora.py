"""Real BERT/Transformers/PEFT lifecycle checks without model downloads."""

from __future__ import annotations

import builtins
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch
import yaml
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file, save_file
from sie_server.adapters.cross_encoder import CrossEncoderAdapter
from sie_server.adapters.lora import LoraStateUncertainError, LoraUnavailableError
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.core.registry import ModelRegistry
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

QUERY = {"text": "how do i reset my password"}
DOCS = [{"text": "reset your password by email"}, {"text": "the capital is paris"}]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cross-encoder-base")
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        *["how", "do", "i", "reset", "my", "password", "your", "by", "email", "the", "capital", "is", "paris"],
    ]
    vocab = list(dict.fromkeys(vocab))
    vocab_path = directory / "vocab.txt"
    vocab_path.write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(vocab_path), model_max_length=64).save_pretrained(directory)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=1,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    config.sbert_ce_default_activation_function = "torch.nn.modules.linear.Identity"
    with torch.random.fork_rng():
        torch.manual_seed(29)
        BertForSequenceClassification(config).save_pretrained(directory)
    return directory


def save_adapter(path, seed=1, **overrides):
    path.mkdir()
    LoraConfig(
        task_type="SEQ_CLS",
        r=4,
        lora_alpha=8,
        target_modules=["query", "value"],
        bias="none",
        lora_dropout=0,
        inference_mode=True,
    ).save_pretrained(path)
    if overrides:
        config_path = path / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config.update(overrides)
        config_path.write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(seed)
    weights = {}
    for layer in range(2):
        for target in ("query", "value"):
            prefix = f"base_model.model.bert.encoder.layer.{layer}.attention.self.{target}"
            weights[f"{prefix}.lora_A.weight"] = torch.randn(4, 16, generator=generator) * 0.4
            weights[f"{prefix}.lora_B.weight"] = torch.randn(16, 4, generator=generator) * 0.4
    save_file(weights, path / "adapter_model.safetensors")
    return str(path)


@pytest.fixture
def adapter(checkpoint, tmp_path):
    paths = {name: save_adapter(tmp_path / name, seed) for seed, name in enumerate(("a", "b", "c"), 1)}
    adapter = CrossEncoderAdapter(checkpoint, lora_paths=paths, max_loaded_loras=2)
    adapter.load("cpu")
    yield adapter
    adapter.unload()


def scores(adapter):
    from sie_server.types.inputs import Item

    query, docs = Item(**QUERY), [Item(**doc) for doc in DOCS]
    direct = np.asarray(adapter.score(query, docs))
    pairs = adapter.score_pairs([query] * len(docs), docs)
    assert np.isfinite(direct).all()
    np.testing.assert_allclose(pairs.scores, direct, atol=1e-7, rtol=0)
    assert pairs.input_token_counts
    assert all(count > 0 for count in pairs.input_token_counts)
    return direct


def base_snapshot(adapter):
    outer = adapter._model
    return (
        outer,
        outer.model,
        outer.model.classifier,
        outer.tokenizer,
        outer.activation_fn,
        dict(outer.model.named_modules()),
        {name: value.detach().clone() for name, value in outer.model.state_dict().items()},
        [(parameter, parameter.requires_grad) for parameter in outer.model.parameters()],
    )


def assert_base_preserved(adapter, snapshot, *, unwrapped=False):
    outer, model, head, tokenizer, activation, modules, weights, grad = snapshot
    assert adapter._model is outer
    assert outer.model is model
    assert model.classifier is head
    assert outer.tokenizer is tokenizer
    assert outer.activation_fn is activation
    for name, expected in weights.items():
        parent, _, leaf = name.rpartition(".")
        module = model.get_submodule(parent)
        if isinstance(module, BaseTunerLayer):
            module = module.get_base_layer()
        assert torch.equal(getattr(module, leaf), expected), name
    assert all(parameter.requires_grad == flag for parameter, flag in grad)
    if unwrapped:
        assert dict(model.named_modules()) == modules
        assert not hasattr(model, "peft_config")
        assert not model._hf_peft_config_loaded


def test_never_lora_has_no_manager_or_optional_imports(checkpoint, monkeypatch):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "sie_server.adapters.cross_encoder._lora"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    adapter = CrossEncoderAdapter(checkpoint)
    assert adapter._lora is None
    assert not adapter.supports_lora()
    assert adapter.lora_capability() is None
    assert not adapter.supports_hot_lora_reload()
    assert adapter.lora_runtime_config() is None
    adapter.load("cpu")
    before = base_snapshot(adapter)
    baseline = scores(adapter)
    adapter.set_active_lora(None)
    np.testing.assert_array_equal(scores(adapter), baseline)
    assert_base_preserved(adapter, before)
    with pytest.raises(ValueError, match="No LoRA sources"):
        adapter.set_active_lora("a")
    adapter.unload()


def test_actual_effect_restoration_head_and_final_module_identity(adapter):
    snapshot = base_snapshot(adapter)
    baseline = scores(adapter)
    np.testing.assert_array_equal(scores(adapter), baseline)
    runtime = adapter.lora_runtime_config()
    assert runtime is not None
    assert runtime.max_loaded == 2
    assert set(runtime.aliases) == {"a", "b", "c"}
    assert not runtime.preloaded
    assert not runtime.pinned
    with pytest.raises(LoraUnavailableError):
        adapter.set_active_lora("a")
    assert adapter.load_lora("a") == 2048
    # Successful attachment must leave base selected, including SEQ_CLS configs.
    np.testing.assert_array_equal(scores(adapter), baseline)
    adapter.set_active_lora("a")
    a = scores(adapter)
    assert not any(parameter.requires_grad for parameter in adapter._lora._parameters[adapter._lora._aliases["a"]])
    assert np.max(np.abs(a - baseline)) > 1e-6
    assert_base_preserved(adapter, snapshot)
    adapter.load_lora("b")
    np.testing.assert_array_equal(scores(adapter), baseline)
    adapter.set_active_lora("b")
    b = scores(adapter)
    assert np.max(np.abs(b - a)) > 1e-6
    adapter.set_active_lora(None)
    np.testing.assert_array_equal(scores(adapter), baseline)
    with pytest.raises(ValueError, match="capacity"):
        adapter.load_lora("c")
    adapter.unload_lora("a")
    with pytest.raises(LoraUnavailableError):
        adapter.set_active_lora("a")
    adapter.set_active_lora("b")
    np.testing.assert_array_equal(scores(adapter), b)
    adapter.unload_lora("b")
    assert_base_preserved(adapter, snapshot, unwrapped=True)
    np.testing.assert_array_equal(scores(adapter), baseline)
    adapter.load_lora("a")
    adapter.set_active_lora("a")
    np.testing.assert_array_equal(scores(adapter), a)


@pytest.mark.parametrize("existing", [False, True])
def test_failed_attachment_rolls_back_entire_base_or_only_failed_adapter(adapter, monkeypatch, existing):
    snapshot = base_snapshot(adapter)
    baseline = scores(adapter)
    if existing:
        adapter.load_lora("a")
        adapter.set_active_lora("a")
        a = scores(adapter)
    original = adapter._model.load_adapter

    def fail_after_injection(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected post-load failure")

    with monkeypatch.context() as context:
        context.setattr(adapter._model, "load_adapter", fail_after_injection)
        with pytest.raises(RuntimeError, match="injected post-load"):
            adapter.load_lora("b")
    assert_base_preserved(adapter, snapshot, unwrapped=not existing)
    np.testing.assert_array_equal(scores(adapter), baseline)
    assert adapter._lora._prepared == {}
    with pytest.raises(LoraUnavailableError):
        adapter.set_active_lora("b")
    if existing:
        adapter.set_active_lora("a")
        np.testing.assert_array_equal(scores(adapter), a)
    adapter.load_lora("b")
    adapter.set_active_lora("b")
    assert np.max(np.abs(scores(adapter) - baseline)) > 1e-6


@pytest.mark.parametrize("damage", ["missing", "empty", "foreign", "shape", "nan", "overflow", "duplicate"])
def test_bad_weights_rejected_before_model_mutation(adapter, damage):
    source = Path(adapter._lora._aliases["a"])
    weights_path = source / "adapter_model.safetensors"
    weights = load_file(weights_path)
    key = next(iter(weights))
    if damage == "missing":
        del weights[key]
    elif damage == "empty":
        weights.clear()
    elif damage == "foreign":
        weights["base_model.model.classifier.weight"] = torch.ones(1, 16)
    elif damage == "shape":
        weights[key] = torch.ones(3, 16)
    elif damage == "nan":
        weights[key][0, 0] = float("nan")
    elif damage == "overflow":
        weights[key] = torch.full(weights[key].shape, 1e300, dtype=torch.float64)
    else:
        weights[key.removeprefix("base_model.model.")] = weights[key].clone()
    save_file(weights, weights_path)
    snapshot = base_snapshot(adapter)
    with pytest.raises(ValueError, match="LoRA"):
        adapter.load_lora("a")
    assert_base_preserved(adapter, snapshot, unwrapped=True)
    assert adapter._lora._prepared == {}
    assert not adapter._lora._loaded


@pytest.mark.parametrize(
    "change",
    [
        {"modules_to_save": ["classifier"]},
        {"bias": "all"},
        {"target_modules": ["classifier"]},
        {"target_modules": ["dense"]},
        {"target_modules": ["does_not_exist"]},
        {"target_modules": "query"},
        {"init_lora_weights": "pissa"},
        {"use_dora": True},
        {"rank_pattern": {"query": 2}},
        {"target_parameters": ["classifier.weight"]},
        {"lora_alpha": float("inf")},
    ],
)
def test_unsafe_or_unqualified_config_is_rejected(adapter, change):
    path = Path(adapter._lora._aliases["a"]) / "adapter_config.json"
    config = json.loads(path.read_text())
    config.update(change)
    path.write_text(json.dumps(config))
    snapshot = base_snapshot(adapter)
    with pytest.raises(ValueError, match="LoRA"):
        adapter.load_lora("a")
    assert_base_preserved(adapter, snapshot, unwrapped=True)


def test_only_declared_private_snapshot_can_attach(adapter, tmp_path):
    unknown = save_adapter(tmp_path / "unknown")
    with pytest.raises(ValueError, match="not declared"):
        adapter.fetch_lora(unknown)
    with pytest.raises(ValueError, match="validated snapshot"):
        adapter.load_lora("a", source=unknown)
    prepared = Path(adapter.fetch_lora("a"))
    assert prepared.is_dir()
    assert prepared != Path(adapter._lora._aliases["a"])
    (Path(adapter._lora._aliases["a"]) / "adapter_model.safetensors").write_bytes(b"corrupted after validation")
    assert adapter.load_lora("a", source=str(prepared)) > 0
    assert not prepared.exists()


def test_uncertain_mutation_refuses_even_base_until_whole_model_unload(adapter, monkeypatch):
    adapter.load_lora("a")
    snapshot = base_snapshot(adapter)

    def fail(*args, **kwargs):
        raise RuntimeError("injected rollback failure")

    with monkeypatch.context() as context:
        context.setattr(adapter._model, "load_adapter", fail)
        context.setattr(adapter._lora, "_remove", fail)
        with pytest.raises(LoraStateUncertainError):
            adapter.load_lora("b")
    with pytest.raises(LoraStateUncertainError):
        adapter.set_active_lora(None)
    adapter.unload()
    assert not any(isinstance(module, BaseTunerLayer) for module in snapshot[1].modules())
    assert adapter._lora._outer is None
    assert adapter._lora._originals == {}
    adapter.load("cpu")
    adapter.load_lora("a")
    adapter.set_active_lora("a")
    assert np.isfinite(scores(adapter)).all()


def test_discard_and_model_unload_release_owned_sources(adapter):
    a = Path(adapter.fetch_lora("a"))
    adapter.discard_lora_source("a")
    assert not a.exists()
    b = Path(adapter.fetch_lora("b"))
    model = adapter._model.model
    adapter.unload()
    assert not b.exists()
    assert not adapter._lora._prepared
    assert not any(isinstance(module, BaseTunerLayer) for module in model.modules())
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.fetch_lora("a")


def test_revision_and_capacity_validation(checkpoint, tmp_path):
    path = save_adapter(tmp_path / "a")
    with pytest.raises(ValueError, match="positive integer"):
        CrossEncoderAdapter(checkpoint, lora_paths={"a": path}, max_loaded_loras=0)
    adapter = CrossEncoderAdapter(checkpoint, lora_paths={"a": path}, lora_revisions={path: "a" * 40})
    adapter.load("cpu")
    try:
        with pytest.raises(ValueError, match="differs"):
            adapter.fetch_lora("a", revision="b" * 40)
        with pytest.raises(ValueError, match="local LoRA"):
            adapter.fetch_lora("a")
    finally:
        adapter.unload()


def test_revisions_alone_do_not_enable_or_authorize_lora(checkpoint):
    adapter = CrossEncoderAdapter(checkpoint, lora_revisions={"other-profile/source": "a" * 40})
    assert adapter._lora is None
    assert adapter.lora_capability() is None


@pytest.mark.parametrize("paths", [["org/source"], {"/private/source": "org/source"}, {"a": "b", "b": "c"}])
def test_aliases_are_public_names_and_cannot_shadow_sources(checkpoint, paths):
    with pytest.raises(ValueError, match=r"aliases|shadows"):
        CrossEncoderAdapter(checkpoint, lora_paths=paths)


def test_fetch_rechecks_model_generation_before_publishing_source(adapter, monkeypatch):
    from sie_server.adapters.cross_encoder import _lora

    original = _lora.load_file

    def reload_during_fetch(path):
        weights = original(path)
        adapter.unload()
        adapter.load("cpu")
        return weights

    monkeypatch.setattr(_lora, "load_file", reload_during_fetch)
    with pytest.raises(RuntimeError, match="reloaded while"):
        adapter.fetch_lora("a")
    assert adapter._lora._prepared == {}


@pytest.mark.parametrize("existing", [False, True])
def test_partial_injection_failure_preserves_old_adapters_and_base(adapter, monkeypatch, existing):
    from peft.tuners.lora.model import LoraModel

    snapshot = base_snapshot(adapter)
    baseline = scores(adapter)
    if existing:
        adapter.load_lora("a")
        adapter.set_active_lora("a")
        a = scores(adapter)
    original = LoraModel._create_and_replace
    calls = 0

    def fail_second_layer(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected mid-attachment failure")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(LoraModel, "_create_and_replace", fail_second_layer)
        with pytest.raises(RuntimeError, match="mid-attachment"):
            adapter.load_lora("b")
    assert calls == 2
    assert_base_preserved(adapter, snapshot, unwrapped=not existing)
    np.testing.assert_array_equal(scores(adapter), baseline)
    if existing:
        adapter.set_active_lora("a")
        np.testing.assert_array_equal(scores(adapter), a)
    adapter.load_lora("b")


def test_source_cleanup_failure_retains_capacity_and_blocks_selection(adapter, monkeypatch):
    path = adapter.fetch_lora("a")
    canonical = adapter._lora._aliases["a"]
    temporary = adapter._lora._prepared[canonical].directory

    def fail_cleanup():
        raise OSError("injected cleanup failure")

    with monkeypatch.context() as context:
        context.setattr(temporary, "cleanup", fail_cleanup)
        with pytest.raises(LoraStateUncertainError, match="cleanup failed"):
            adapter.load_lora("a", source=path)
    assert canonical in adapter._lora._loaded
    assert canonical in adapter._lora._prepared
    with pytest.raises(LoraStateUncertainError):
        adapter.set_active_lora(None)
    adapter.unload()
    assert not Path(path).exists()


def test_source_cleanup_failure_cannot_skip_whole_model_teardown(adapter, monkeypatch):
    adapter.load_lora("a")
    original_model = adapter._model.model
    path = adapter.fetch_lora("b")
    temporary = adapter._lora._prepared[adapter._lora._aliases["b"]].directory

    def fail_cleanup():
        raise OSError("injected cleanup failure")

    with monkeypatch.context() as context:
        context.setattr(temporary, "cleanup", fail_cleanup)
        with pytest.raises(ExceptionGroup, match="owned LoRA"):
            adapter.unload()
    assert adapter._model is None
    assert adapter._lora._model is None
    assert not adapter._lora._loaded
    assert not any(isinstance(module, BaseTunerLayer) for module in original_model.modules())
    assert Path(path).exists()
    temporary.cleanup()


@pytest.mark.usefixtures("_offline_apps")
def test_real_score_routes_authorize_aliases_isolate_batches_and_evict(
    checkpoint,
    tmp_path,
    monkeypatch,
    serve_on_loopback,
):
    from sie_server.adapters.cross_encoder._lora import CrossEncoderLora

    paths = {name: save_adapter(tmp_path / name, index) for index, name in enumerate(("a", "b", "c", "other"), 1)}
    models = tmp_path / "models"
    models.mkdir()
    config = {
        "sie_id": "test/cross-encoder-lora",
        "weights_path": str(checkpoint),
        "inputs": {"text": True},
        "tasks": {"score": {}},
        "max_sequence_length": 64,
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.cross_encoder:CrossEncoderAdapter",
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": {"lora_paths": {name: paths[name] for name in ("a", "b", "c")}}},
            },
            "other": {"extends": "default", "adapter_options": {"loadtime": {"lora_paths": {"other": paths["other"]}}}},
        },
    }
    (models / "model.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("SIE_MAX_LORAS_PER_MODEL", "2")
    monkeypatch.setenv("SIE_DISK_CACHE_ENABLED", "false")

    async def no_watcher(self):
        pass

    monkeypatch.setattr(ModelRegistry, "start_hot_reload", no_watcher)
    fetched = []
    original_fetch = CrossEncoderLora.fetch

    def observed_fetch(self, name, revision=None):
        fetched.append(name)
        return original_fetch(self, name, revision)

    monkeypatch.setattr(CrossEncoderLora, "fetch", observed_fetch)
    app = AppFactory.create_app(AppStateConfig(models_dir=str(models), device="cpu", remote_serving=False))
    model = config["sie_id"]
    with (
        serve_on_loopback(app) as url,
        httpx.Client(base_url=url, timeout=15, headers={"Accept": "application/json"}) as client,
    ):

        def request(selection=None, *, profile=None):
            options = {"lora": selection}
            if profile is not None:
                options["profile"] = profile
            return client.post(f"/v1/score/{model}", json={"query": QUERY, "items": DOCS, "options": options})

        def ready(selection=None):
            deadline = time.monotonic() + 20
            response = request(selection)
            statuses = [response.status_code]
            if selection is not None and response.status_code == 503:
                assert response.json()["detail"]["code"] == "LORA_LOADING"
                assert float(response.headers["Retry-After"]) > 0
            while response.status_code == 503 and time.monotonic() < deadline:
                time.sleep(0.02)
                response = request(selection)
                statuses.append(response.status_code)
            assert response.status_code == 200, response.text
            values = np.asarray(
                [entry["score"] for entry in sorted(response.json()["scores"], key=lambda row: row["item_id"])]
            )
            assert np.isfinite(values).all()
            return values, statuses

        baseline, _ = ready()
        assert fetched == []
        for rejected, profile in (
            ("https://example.invalid/adapter", None),
            (str(tmp_path / "not-declared"), None),
            ("other", None),
            (paths["other"], None),
            ("a", "other"),
        ):
            response = request(rejected, profile=profile)
            assert response.status_code == 400, response.text
            assert response.json()["detail"]["code"] == "INVALID_INPUT"
            assert fetched == []
        a, statuses = ready("a")
        assert statuses[0] == 503
        b, statuses = ready("b")
        assert statuses[0] == 503
        assert np.max(np.abs(a - baseline)) > 1e-6
        assert np.max(np.abs(b - a)) > 1e-6
        assert set(app.state.registry.loaded_lora_names(model)) == {"a", "b"}
        listed = client.get("/v1/models").json()["models"]
        metadata = next(item for item in listed if item["name"] == model)
        assert set(metadata["lora_adapters"]) == {"a", "b"}
        assert not any(source in json.dumps(metadata["lora_adapters"]) for source in paths.values())
        assert len(app.state.registry._loaded[model].loras) == 2
        assert all(value.memory_bytes == 2048 for value in app.state.registry._loaded[model].loras.values())
        expected = {None: baseline, "a": a, "b": b}
        selections = ["a", None, "b"] * 8
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [(selection, executor.submit(ready, selection)) for selection in selections]
            for selection, future in futures:
                actual, statuses = future.result(timeout=20)
                assert statuses == [200]
                np.testing.assert_allclose(actual, expected[selection], atol=1e-6, rtol=0)
        ready("b")  # A is now the least recently used resident.
        ready("c")
        assert set(app.state.registry.loaded_lora_names(model)) == {"b", "c"}
        assert not app.state.registry.is_lora_loaded(model, "a")
        np.testing.assert_allclose(ready()[0], baseline, atol=1e-6, rtol=0)
        assert fetched == [paths["a"], paths["b"], paths["c"]]
