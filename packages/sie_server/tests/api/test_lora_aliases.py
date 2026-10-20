"""Native HTTP/MessagePack ingress rejects selectors before routing or loading."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_sdk._msgpack import packb
from sie_server.api import encode, extract, score
from sie_server.config.model import ModelConfig
from sie_server.core.registry import ModelRegistry


@pytest.mark.parametrize("endpoint", ["encode", "extract", "score"])
@pytest.mark.parametrize("wire", ["json", "msgpack"])
@pytest.mark.parametrize(
    "options",
    [
        {"lora_adapter": None, "lora": "org/legal"},
        {"lora_adapter": "org/legal", "lora_id": "other"},
        {"lora_adapter": []},
        {"lora_adapter": " "},
        {"lora_id": "legal"},
        {"lora_adapter": "org/undeclared"},
    ],
)
def test_bad_lora_options_stop_before_routing_loading_or_worker(endpoint, wire, options, monkeypatch):
    config = ModelConfig.model_validate(
        {
            "sie_id": "m",
            "hf_id": "unloaded",
            "inputs": {"text": True},
            "tasks": {"encode": {"dense": {"dim": 8}}, "extract": {}, "score": {}},
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter",
                    "adapter_options": {"loadtime": {"lora_paths": {"legal": "org/legal"}}},
                }
            },
        }
    )
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = registry.is_loaded.return_value = True
    registry.is_loading.return_value = registry.is_unloading.return_value = registry.is_failed.return_value = False
    registry.get_config.return_value = config
    registry.device = "cpu"
    app = FastAPI()
    module = {"encode": encode, "extract": extract, "score": score}[endpoint]
    app.include_router(module.router)
    app.state.registry = registry
    route = AsyncMock(side_effect=AssertionError("bad selection reached route"))
    monkeypatch.setattr(module, "route_request", route)
    body = {"items": [{"text": "x"}], "params": {"labels": ["label"], "options": options}}
    if endpoint == "score":
        body = {"query": {"text": "q"}, "items": [{"text": "x"}], "options": options}
    with TestClient(app) as client:
        if wire == "json":
            response = client.post(f"/v1/{endpoint}/m", json=body)
        else:
            response = client.post(
                f"/v1/{endpoint}/m", content=packb(body), headers={"content-type": "application/msgpack"}
            )
    assert response.status_code == 400, response.text
    route.assert_not_awaited()
    registry.ensure_lora_loaded_async.assert_not_called()
    registry.start_worker.assert_not_called()
