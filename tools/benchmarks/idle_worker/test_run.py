"""The timing probe follows the SDK's real extraction transport path."""

import httpx
from sie_sdk import SIEClient
from sie_sdk._msgpack import packb
from sie_sdk.types import Item

from tools.benchmarks.idle_worker.run import timing_sample


def test_timing_hook_follows_sdk_extract_route() -> None:
    samples = []

    def response_for(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={
                "X-Queue-Time": "3.81",
                "X-Inference-Time": "20.0",
                "X-Total-Time": "24.0",
                "Content-Type": "application/msgpack",
            },
            content=packb(
                {
                    "model": "knowledgator/gliclass-large-v1.0",
                        "items": [
                        {"id": "probe", "entities": [], "classifications": [{"label": "billing", "score": 0.9}]}
                    ],
                }
            ),
        )

    transport = httpx.Client(
        base_url="http://127.0.0.1:8373",
        transport=httpx.MockTransport(response_for),
        event_hooks={"response": [lambda response: samples.append(timing_sample(response))]},
    )
    with SIEClient("http://127.0.0.1:8373", http_client=transport) as client:
        client.extract("knowledgator/gliclass-large-v1.0", Item(text="Refund please", id="probe"), labels=["billing"])
    assert samples == [{"queue_ms": 3.81, "inference_ms": 20.0, "total_ms": 24.0}]
