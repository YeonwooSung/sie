# Runtime LoRA acceptance probe

This standalone probe tests SGLang runtime LoRA through `sie_sdk.SIEClient` on
one CUDA device. It runs pinned Qwen3-0.6B generation and Qwen3-Embedding-0.6B
embedding engines sequentially, each with `mem_fraction_static: 0.5`.
It requires the runtime LoRA implementation and SGLang `0.5.10.post1`.

The local fixtures are **untrained correctness fixtures**, not useful task
adapters. Fixed-seed, nonzero rank-4 `q_proj`/`v_proj` weights make incorrect
selection observable without depending on a mutable Hub adapter. No quality
or performance conclusions follow from their outputs.

From a public checkout with its dependencies installed:

```sh
mise run test -- tools/benchmarks/runtime_lora
mise exec -- uv run --no-sync python -m tools.benchmarks.runtime_lora.fixtures \
  --output /tmp/runtime-lora-fixtures --download-weights
mise exec -- uv run --no-sync python -m tools.benchmarks.runtime_lora.probe \
  --fixtures /tmp/runtime-lora-fixtures --output /tmp/runtime-lora-evidence
```

For an installed CUDA bundle, run the same Python modules in that bundle's
environment; do not overlay an incompatible dependency environment. The CPU
tests verify fixture loading with PEFT, base restoration, SDK retry observation,
configuration, and observer behavior. **CPU tests are not GPU acceptance.**

The acceptance run checks:

- Actual `503 LORA_LOADING` followed by another successful SDK request, with no
  direct engine calls substituted for native generation, chat, or embeddings.
  Encode retries automatically. The current generation SDK surfaces this
  refusal, so the probe records and makes bounded fresh SDK calls only for
  pre-execution `503 LORA_LOADING`; it never retries a failed open stream.
- Adapter effects in chat token log probabilities and embedding vectors; base
  restoration after switching and after eviction. Pairwise effects must exceed
  the fixed minimum and five times measured base-repeat drift. These untrained
  fixtures provide selection evidence, not model-quality evidence.
- Stable engine PID, rejection of undeclared adapters, ready-name listing,
  preservation of one pinned adapter, and churn across four dynamic adapters.
- Overlapping A/B streams **and a successful engine forward containing both**.
  A benchmark-only subclass prepends an observer to the generation child's
  startup hooks, preserving SIE's normal compatibility hooks. The observer
  records request IDs, adapter UUIDs and aliases, and actual GPU pool slot
  assignments after `Scheduler.run_batch`. The A/B request UUIDs must occupy
  distinct valid slots, with consistent inverse mappings. Completed SDK streams
  establish completion; the hook timestamp is not a GPU timing measurement.
  It also requires SIE's slot-release compatibility hook to be present.
  Observation does not alter scheduling or model tensors.

`evidence.json` contains the final verdict. `events.jsonl`, `batches.jsonl`,
fixture hashes, generated model configurations, and `source-manifest.json`
provide the supporting record. Use a fresh output directory. The run has a
780-second default deadline and unloads each owned engine before starting the
next; any external compute allocation must have its own finite lifetime and
be stopped and verified separately by its owner.

Pinned primary sources: [generation config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json),
[embedding config](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config.json),
[SGLang scheduler](https://github.com/sgl-project/sglang/blob/v0.5.10.post1/python/sglang/srt/managers/scheduler.py),
and [LoRA GPU pool](https://github.com/sgl-project/sglang/blob/v0.5.10.post1/python/sglang/srt/lora/mem_pool.py).
