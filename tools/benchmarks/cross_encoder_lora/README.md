# CrossEncoder runtime LoRA qualification

This probe measures the ordinary `CrossEncoderAdapter` using the real SIE loader,
worker, loopback HTTP API and `sie_sdk.SIEClient`. It does not provision compute,
install dependencies, change precision, or change the production catalog. The
catalog model is overridden locally to use the ordinary adapter rather than the
catalog's BertFlash implementation.

The fixed checkpoint is `cross-encoder/ms-marco-MiniLM-L-6-v2` at
`c5ee24cb16019beea0893ab7796b1df96625c6b8`; `model.safetensors` must hash to
`821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`.
Fixtures A/B/C are deterministic, untrained rank-4, alpha-8, bias-free LoRAs on
every BERT encoder query/value projection. They establish reversible numerical
effects and lifecycle behavior, **not retrieval quality**. Source seeds, model
files and adapter files are hashed in the fixture manifest.

## Prepare immutable sources and fixtures

From a clean public checkout with the candidate committed:

```console
mise tasks
mise exec -- uv run --no-sync python -m tools.benchmarks.cross_encoder_lora.prepare --candidate HEAD --output /tmp/ce-sources
mise exec -- uv run --no-sync python -m tools.benchmarks.cross_encoder_lora.fixtures --output /tmp/ce-fixtures
```

The preparer resolves refs once, archives baseline
`d5e024ad3ba6127b1ee176e2c85c794b76077888` and the candidate by full commit ID,
and refuses differing dependency locks. Keep the archives with their manifest.
The same benchmark code observes every arm; `PYTHONPATH` selects the archived
server and SDK. No arm runs a dependency sync. Pass `--snapshot` to both fixture
preparation and the runner when using a retained public checkpoint directory;
the worker checks its full recorded file manifest before loading it.

Prepare one Linux L4 environment using the public lock. The qualification
preflight requires Torch `2.9.1+cu129`, Transformers `4.57.6`,
SentenceTransformers `5.4.1`, PEFT `0.19.1`, and **no installed `torchao`**.
Four Torch intra-op threads and one inter-op thread are fixed. Default CUDA
precision remains unchanged (fp16 for this adapter). Record any environment
preparation separately, including installed versions before and after; the
benchmark only inspects them and fails on a mismatch.

```console
mise exec -- uv run --no-sync python -m tools.benchmarks.cross_encoder_lora.runner \
  --inputs /tmp/ce-sources --fixtures /tmp/ce-fixtures \
  --output /tmp/ce-evidence --deadline-s 1740 --trial-deadline-s 120
```

Use a fresh output directory. Set the outer compute-session deadline before
launching, allowing bounded startup and cleanup. The runner owns and joins only
its subprocesses; it cannot terminate an external GPU session. The operator
must stop that exact session and record cleanup. A failed or incomplete trial
stops the experiment and retains its evidence. Never change thresholds or drop
failed trials to obtain a pass.

## Frozen comparison

There are six fresh-process blocks, in order `BNR`, `NRB`, `RBN`, `BRN`, `RNB`,
`NBR` (18 fresh servers). B uses the pinned baseline with no declared LoRA.
N uses the candidate with no declared LoRA. R uses the candidate after A/B
activity, with base selected and two adapters resident but disabled.

Every arm uses the same password-reset query and four documents in `protocol.py`.
Each phase starts with 30 sequential warmup requests. C1 measures 200 sequential
requests; C4 and C16 each run a 12-second closed loop with one SDK client per
lane. Throughput includes draining requests admitted before the deadline.
Latencies use the nearest-rank p95. Neither warmup nor correctness calls enter
the primary statistics.

All six metrics are required for **both** N/B and R/B: C1 mean and p95 latency,
C4/C16 requests per second and p95 latency. The statistical unit is the six
paired blocks. Latency ratios are candidate/B; throughput ratios are B/candidate.
For log ratios, the one-sided 95% t upper bound (df=5, critical
`2.0150483733330233`) must be at most 1.03, the geometric point ratio at most
1.01, and every paired ratio at most 1.10. A wide interval is inconclusive.

Before selection, three SDK base results must be finite and stable at absolute
tolerance `1e-6`. The public fp32 checkpoint oracle has a separately fixed `0.1`
CUDA-fp16 compatibility bound. Every ordered batch shape of 4, 8, …, 64 pairs
has two preselection oracle runs, stable at `1e-6`. Actual batches must match
the corresponding ordered-workload fingerprint and shape; unexpected partial
requests or ordering fail qualification. All candidate base oracles and observed
untimed SDK replay outputs must match the corresponding B-block oracle at `1e-6`. This is
deliberately strict and is never relaxed after measurement.

R also checks cold `503 LORA_LOADING`, bounded fresh SDK retries, duplicate-load
single-flight, undeclared-source rejection before fetch, distinct A/base/B
effects, base restoration, real concurrent base/A/B isolation and same-selection
cross-request batching. It records actual PEFT selection at model forward
boundaries. Mutation cannot overlap a score forward. Base tensors and classifier,
tokenizer and outer-wrapper identities remain unchanged. Churn exceeds the
two-adapter capacity; final removal restores the original Linear objects,
module tree and PEFT flags; same-instance reload and full model unload follow.
Production fault-injection tests separately cover malformed artifacts and
partial-attachment rollback; this probe does not pretend to reproduce them on
every GPU trial.

The hook records deep selection/logit details during correctness checks and
two observed SDK replay rounds at C1/C4/C16. During timed phases the forward
hook is removed and production `score_pairs` and `set_active_lora` are bound
directly. The harness asserts that no observer activity occurred. Response
validation and header decoding happen after the phase clock stops; exactly one
successful HTTP response is required per warmup/measured SDK call, excluding
hidden retries. The timed client retains timer, result-list and raw-response
list-append overhead; this is an SDK workload measurement.

Actual timed C4/C16 batch shapes are **unknown**. Their raw four-score vectors
are compared diagnostically against complete preselection oracle vectors at
`1e-6`. Each response records one deterministic matching oracle reference, or
null if none matches. This witness does not identify the actual shape or imply
a unique match: the retained raw vector and complete oracle corpus allow every
match and its ambiguity to be recomputed. A diagnostic cannot combine
different oracle shapes per element or substitute for the separately observed
shape-qualified replays. Timed C1 has one outstanding four-document request and
must still match its preselection result at `1e-6`. This measurement is scoped to
the observed workload and environment; it does not establish zero overhead or
support for other CrossEncoder architectures.

After synchronization and GC, unadapted allocated GPU bytes must match B.
R's additional allocation is bounded by measured unique adapter storage bytes
rounded up to 512 bytes per storage. Final adapter removal must restore the
initial allocated bytes. Reserved allocator memory, RSS, load duration and
GPU clock/temperature/utilization are recorded separately, not conflated with
live parameter allocation. The memory reference is captured after finite
preselection base and shape-oracle warmup, before any LoRA attachment, to keep
one-time CUDA library initialization separate from adapter storage.

These observation changes are an explicit **preallocation protocol amendment**
to the original plan of observing every timed forward. All trial orders,
warmup/request counts, phase durations, numerical tolerances and statistical
gates remain unchanged. The original timed batch-attribution claim is withdrawn;
the observed untimed replays supply that evidence instead.

## Evidence and CPU checks

`summary.json` schema 2 retains trial outcomes, aggregate metrics, statistical
gates and source hashes. Each entry links its complete `trial.json` by relative
path and SHA-256; raw request arrays and oracle corpora stay in those per-trial
files instead of being copied into the growing summary. Final cross-process
checks reopen the complete records and verify their hashes and identities.
The summary also hashes the complete benchmark file manifest. Each trial
captures the full installed-distribution manifest; its digest must remain
identical across arms. An optional `--environment-receipt` attaches a separate
operator preparation receipt and its SHA-256 (never include credentials).
Each trial contains incremental `events.jsonl`, atomic `trial.json`, generated
model configuration and a separate process log. It preserves raw request
scores, ranks, IDs, usage, latencies, errors, batch membership, logits, tensor
hashes, library versions and shutdown status. Failure paths retain completed
phases and partial request samples. Hash evidence files when archiving them;
the runner records each terminal trial file hash.

The compact representation follows an incomplete run whose stdout archive was
truncated. Its old evidence stays separate and is not acceptance evidence. This
change affects only post-measurement diagnostics and storage, not the timed
path, workload, numerical tolerances, statistical gates, or raw observations.
Store the evidence directory on durable storage and verify the full archive
hash after retrieval; stdout is unsuitable for transferring large artifacts.

```console
mise run test -- tools/benchmarks/cross_encoder_lora/tests/test_probe.py
mise run typecheck -- tools/benchmarks/cross_encoder_lora
```

The CPU suite serves a tiny real BERT model through the actual loader and SDK,
uses real PEFT tensors, and checks correctness plus positive and negative
statistical/provenance controls. Its deliberately shorter phases and CPU
precision are labeled `qualification: false`; it is a harness smoke test,
not acceptance evidence for the pinned MiniLM checkpoint or L4 performance.
