"""CPU contract smoke; these checks do not establish CUDA acceptance."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from peft import PeftModel
from safetensors.torch import load_file
from sie_sdk._msgpack import packb as pack_msgpack
from transformers import Qwen3Config, Qwen3ForCausalLM

from tools.benchmarks.runtime_lora.fixtures import MODELS, digest, write_adapter
from tools.benchmarks.runtime_lora.observer import patch_scheduler
from tools.benchmarks.runtime_lora.probe import ROOT, Client, Evidence, batch_rows, check_effect, configs, mixed_batches


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )


def test_fixture_weights_load_in_peft_change_logits_and_restore_base(tmp_path: Path) -> None:
    config = tiny_config()
    first = write_adapter(tmp_path / "a", config.to_dict(), "tiny-qwen", "test", 17)
    write_adapter(tmp_path / "same", config.to_dict(), "tiny-qwen", "test", 17)
    write_adapter(tmp_path / "b", config.to_dict(), "tiny-qwen", "test", 18)
    assert first["files"]["adapter_model.safetensors"] == digest(tmp_path / "same/adapter_model.safetensors")
    tensors = load_file(tmp_path / "a/adapter_model.safetensors")
    assert len(tensors) == 8
    assert all(torch.count_nonzero(value) == value.numel() for value in tensors.values())
    torch.manual_seed(1)
    model = Qwen3ForCausalLM(config).eval()
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        base = model(inputs).logits.clone()
    model = PeftModel.from_pretrained(model, tmp_path / "a", adapter_name="a")
    model.load_adapter(tmp_path / "b", adapter_name="b")
    with torch.no_grad():
        model.set_adapter("a")
        a = model(inputs).logits.clone()
        model.set_adapter("b")
        b = model(inputs).logits.clone()
        with model.disable_adapter():
            restored = model(inputs).logits
    assert torch.max(torch.abs(base - a)) > 1e-5
    assert torch.max(torch.abs(base - b)) > 1e-5
    assert torch.max(torch.abs(a - b)) > 1e-5
    torch.testing.assert_close(restored, base, rtol=0, atol=1e-6)


def test_catalog_pins_and_declared_fixture_config(tmp_path: Path) -> None:
    generated = configs(tmp_path / "models", tmp_path / "fixtures")
    for kind, (name, revision) in MODELS.items():
        data = generated[kind]
        assert data["hf_id"] == name
        assert data["hf_revision"] == revision
        options = data["profiles"]["default"]["adapter_options"]["loadtime"]
        assert options["preload_loras"] == ["pin"]
        assert set(options["lora_paths"]) == {"pin", "a", "b", "c", "d"}
        assert options["max_loras_per_batch"] == 2
        assert options["max_loaded_loras"] == 3
        assert options["mem_fraction_static"] == 0.5


def test_sdk_observer_captures_cold_503_and_native_retry(tmp_path: Path) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                503, json={"error": {"code": "LORA_LOADING", "message": "loading"}}, headers={"Retry-After": "0"}
            )
        return httpx.Response(200, json={"model": MODELS["generation"][0], "text": "ok"})

    client = Client("http://localhost", Evidence(tmp_path), transport=httpx.MockTransport(respond))
    try:
        assert client.call("generation", MODELS["generation"][0], "a", "generate") == {"text": "ok"}
        assert [row["status"] for row in client.responses] == [503, 200]
        assert all(request["lora_adapter"] == "a" for request in calls)
        assert client.responses[0]["code"] == "LORA_LOADING"
    finally:
        client.close()


@pytest.mark.parametrize("error_content_type", ["application/json", "application/msgpack"])
def test_sdk_observer_captures_msgpack_selection_and_automatic_encode_retry(
    tmp_path: Path, error_content_type: str
) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            error = {"error": {"code": "LORA_LOADING", "message": "loading"}}
            content = pack_msgpack(error) if error_content_type == "application/msgpack" else json.dumps(error).encode()
            return httpx.Response(
                503, content=content, headers={"Content-Type": error_content_type, "Retry-After": "0"}
            )
        return httpx.Response(
            200,
            content=pack_msgpack({"items": [{"dense": {"dims": 2, "values": np.array([0.5, 0.25])}}] * 2}),
            headers={"Content-Type": "application/msgpack"},
        )

    client = Client("http://localhost", Evidence(tmp_path), transport=httpx.MockTransport(respond))
    try:
        assert client.call("embedding", MODELS["embedding"][0], "a") == {"vectors": [[0.5, 0.25]] * 2}
        assert [row["status"] for row in client.responses] == [503, 200]
        assert all(row["selection"] == "a" for row in client.responses)
        assert client.responses[0]["code"] == "LORA_LOADING"
        assert all(request.headers["content-type"] == "application/msgpack" for request in calls)
        assert "fresh_sdk_generation_retry" not in (tmp_path / "events.jsonl").read_text()
    finally:
        client.close()


def test_observer_records_actual_successful_batch_and_does_not_change_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "batches.jsonl"
    result = object()
    seen = []

    class Scheduler:
        def run_batch(self, batch, extra=None):
            seen.append((batch, extra))
            if extra == "fail":
                raise ValueError("forward failed")
            return result

    module = ModuleType("fake_scheduler")
    module.Scheduler = Scheduler
    patch_scheduler(module, str(trace))
    scheduler = Scheduler()
    uid_a, uid_b = "62f6770ac6ac4f718ff17fe577616e01", "ad52764f1b98498595d08d1243932632"
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            lora_manager=SimpleNamespace(
                lora_refs={uid_a: SimpleNamespace(lora_name="a"), uid_b: SimpleNamespace(lora_name="b")},
                memory_pool=SimpleNamespace(buffer_id_to_uid=[uid_a, uid_b], uid_to_buffer_id={uid_a: 0, uid_b: 1}),
            )
        )
    )
    batch = SimpleNamespace(
        forward_mode="DECODE", reqs=[SimpleNamespace(rid="1", lora_id=uid_a), SimpleNamespace(rid="2", lora_id=uid_b)]
    )
    assert scheduler.run_batch(batch, extra=3) is result
    assert seen == [(batch, 3)]
    with pytest.raises(ValueError, match="forward failed"):
        scheduler.run_batch(batch, extra="fail")
    rows = batch_rows(trace)
    assert len(rows) == 1
    assert rows[0]["gpu_slot_count"] == 2
    assert rows[0]["requests"][0] == {"rid": "1", "uid": uid_a, "lora": "a"}
    assert rows[0]["gpu_slots"] == [{"uid": uid_a, "lora": "a", "slot": 0}, {"uid": uid_b, "lora": "b", "slot": 1}]
    assert mixed_batches(rows, 0, 2**64) == rows
    rows[0]["gpu_slot_uids"][1] = uid_a
    assert mixed_batches(rows, 0, 2**64) == []
    rows[0]["gpu_slot_uids"][1] = uid_b
    rows[0]["gpu_slots"][1]["slot"] = 0
    assert mixed_batches(rows, 0, 2**64) == []
    rows[0]["gpu_slots"][1]["slot"] = 1
    rows[0]["requests"][1]["lora"] = "a"
    assert mixed_batches(rows, 0, 2**64) == []
    pool_module = ModuleType("sglang.srt.lora.mem_pool")
    pool_module.EMPTY_SLOT = object()
    monkeypatch.setitem(sys.modules, pool_module.__name__, pool_module)
    pool = scheduler.tp_worker.model_runner.lora_manager.memory_pool
    pool.buffer_id_to_uid = [None, pool_module.EMPTY_SLOT]
    pool.uid_to_buffer_id = {None: 0}
    batch.reqs = [SimpleNamespace(rid="base", lora_id=None)]
    assert scheduler.run_batch(batch) is result
    base_row = batch_rows(trace)[-1]
    assert base_row["gpu_slot_uids"] == [None, {"empty_slot": True}]
    assert base_row["gpu_slots"] == [{"lora": None, "slot": 0, "uid": None}]
    assert mixed_batches([base_row], 0, 2**64) == []


def test_child_observer_defers_import_and_preserves_production_hook(tmp_path: Path) -> None:
    """A synthetic upstream package checks startup composition without CUDA."""
    for package in ("sglang", "sglang/srt", "sglang/srt/managers", "sglang/srt/lora"):
        directory = tmp_path / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()
    metadata = tmp_path / "sglang-0.5.10.post1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: sglang\nVersion: 0.5.10.post1\n")
    # The real slot-release implementation has its own production tests. This
    # sentinel exercises the observer's explicit prerequisite and evidence.
    (tmp_path / "sglang/srt/lora/lora_manager.py").write_text(
        "class LoRAManager:\n    _sie_lora_slot_release_compat = True\n"
    )
    (tmp_path / "sglang/srt/managers/scheduler.py").write_text(
        "from sglang.srt.lora.lora_manager import LoRAManager\n"
        "class Scheduler:\n    def run_batch(self, batch):\n        return 47\n"
    )
    trace = tmp_path / "child-batches.jsonl"
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            map(str, (ROOT / "tools/benchmarks/runtime_lora/_hooks", tmp_path, ROOT, ROOT / "packages/sie_server/src"))
        ),
        "SIE_RUNTIME_LORA_BATCH_TRACE": str(trace),
        "SIE_SGLANG_MM_PROCESS_CONFIG_COMPAT": "1",
        "SIE_SGLANG_LORA_COMPAT": "",
    }
    script = (
        "import sys\nassert 'sglang' not in sys.modules\n"
        "from sglang.srt.managers.scheduler import Scheduler\n"
        "import builtins\nfrom types import SimpleNamespace as S\n"
        "assert getattr(builtins.__import__, '_sie_mm_process_config_deferred_compat', False)\n"
        "scheduler = Scheduler()\n"
        "scheduler.tp_worker = S(model_runner=S(lora_manager=S(lora_refs={}, "
        "memory_pool=S(buffer_id_to_uid=[None], uid_to_buffer_id={None: 0}))))\n"
        "assert scheduler.run_batch(S(forward_mode='DECODE', reqs=[S(rid='r', lora_id=None)])) == 47\n"
    )
    result = subprocess.run(  # noqa: S603 — fixed synthetic package and script, no external input
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    rows = batch_rows(trace)
    assert [row["event"] for row in rows] == ["observer_installed", "forward"]
    assert rows[0]["slot_release_compat"] is True
    assert rows[1]["requests"] == [{"rid": "r", "uid": None, "lora": None}]


def test_effect_checks_fail_for_ignored_adapters_and_contaminated_base() -> None:
    base = {"vectors": [[0.1, 0.2]]}
    a = {"vectors": [[0.2, 0.3]]}
    b = {"vectors": [[0.3, 0.4]]}
    assert check_effect("embedding", base, a, b, base)["base_reset_max_abs"] == 0
    with pytest.raises(AssertionError, match="did not differ"):
        check_effect("embedding", base, base, b, base)
    with pytest.raises(AssertionError, match="not restored"):
        check_effect("embedding", base, a, b, a)
    assert np.isfinite(check_effect("embedding", base, a, b, base)["a_vs_b_max_abs"])


def test_effect_checks_require_delta_beyond_observed_numeric_drift() -> None:
    def embedding(value: float) -> dict:
        return {"vectors": [[value]]}

    def generation(value: float) -> dict:
        return {"text": "same", "logprobs": {"content": [{"token": "same", "logprob": math.log(value)}]}}

    for kind, build, a, b, reset in (
        ("embedding", embedding, 0.0001, 0.0003, 0.00004),
        ("generation", generation, 0.0002, 0.0004, 0.00005),
    ):
        base = build(0.5)
        assert check_effect(kind, base, build(0.5 + a), build(0.5 + b), base)
        with pytest.raises(AssertionError, match="beyond base drift"):
            check_effect(kind, base, build(0.5 + a), build(0.5 + b), build(0.5 + reset))
