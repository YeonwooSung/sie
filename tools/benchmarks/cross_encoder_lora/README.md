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
  --output /tmp/ce-evidence --deadline-s 4200 --trial-deadline-s 270
```

Use a fresh output directory. Set the outer compute-session deadline before
launching, allowing bounded startup and cleanup. The runner owns and joins only
its subprocesses; it cannot terminate an external GPU session. The operator
must stop that exact session and record cleanup. A failed or incomplete trial
stops the experiment and retains its evidence. Each new trial requires strictly
more than its 270-second limit plus 90 seconds for wait, termination and kill
grace (45/35/10). The proposed outer limits are 4,500 seconds for the operator
and 4,800 seconds from provider start, with a 300-second startup cutoff and an
additional absolute ceiling of application creation plus 5,100 seconds. The
operator must reserve export time before starting the runner; these settings
do not themselves authorize or allocate compute. Never change thresholds or
drop failed trials to obtain a pass, or retry the same failed protocol unchanged.

## Frozen comparison

There are six fresh-process blocks, in order `BNR`, `NRB`, `RBN`, `BRN`, `RNB`,
`NBR` (18 fresh servers). B uses the pinned baseline with no declared LoRA.
N uses the candidate with no declared LoRA. R uses the candidate after A/B
activity, with base selected and two adapters resident but disabled.

Every arm uses the same password-reset query and four documents in `protocol.py`.
Each phase starts with 30 sequential warmup requests. C1 measures 2,000 sequential
requests; C4 and C16 each run a 60-second closed loop with one SDK client per
lane. Throughput includes draining requests admitted before the deadline.
Latencies use the nearest-rank p95. Neither warmup nor correctness calls enter
the primary statistics. Concurrent futures have a 120-second wait limit: the
60-second measurement, the existing 30-second request timeout, and 30 seconds
of scheduling margin. The worker and outer process limits still bound failures.

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

The untimed observer wraps the model's bound `forward` method, because Sentence
Transformers calls that method directly and bypasses root module forward hooks.
Every observed outer score batch must have nonempty forward records, finite
logits and exact input-row coverage; selected/base tuner checks cannot pass on
an empty list. It records selection/logit details during correctness checks and
two observed SDK replay rounds at C1/C4/C16. During timed phases the original
forward attribute lookup is restored: an original instance override is restored
exactly, or the temporary instance attribute is deleted to restore descriptor
lookup. Production `score_pairs` and `set_active_lora` are bound directly.
The harness checks the original forward target and that no observer activity
occurred, including restoration after an exceptional exit. Successful response
status and timing headers are captured as small metadata snapshots in the HTTP
observer, without retaining the successful HTTPX response. Error responses stay
available for deferred body and retry evidence. Validation happens after the
phase clock stops; exactly one successful HTTP response is required per
warmup/measured SDK call, excluding hidden retries. The timed client retains
timer, result-list and snapshot overhead; this is an SDK workload measurement.

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

Removing timed forward observation was an earlier **preallocation protocol
amendment**. It preserved that version's trial order, workload and gates, and
withdrew its timed batch-attribution claim. Observed untimed replays supply that
evidence instead.

## Longer-window amendment

The original qualification used 200 C1 requests and 12-second C4/C16 phases.
Its failed full-run evidence and the failed baseline-only stability diagnostic
remain separate, with their original source and artifact hashes. Those outcomes
are not reinterpreted as passes. The diagnostic's longer warmup did not establish
stability, so qualification retains 30 sequential warmup requests.

The revised workload uses 2,000 C1 requests and 60-second concurrent phases to
observe longer intervals. The six balanced fresh-process triplets, checkpoint,
precision, dependencies, request payloads, numerical tolerances and all twelve
statistical gates stay fixed. Longer windows may reduce within-window sampling
variation; they do not guarantee comparability across fresh processes. There
are still only six independent paired units, not thousands of independent
latency samples for the t interval.

A model-free lifetime probe separately established that the old observer held
successful responses for the whole phase and kept its client alive through a
bound callback. Metadata snapshots and removal of the owned hook/history on
close address that benchmark-owned retention. Normal HTTPX response-stream and
SDK namespace cycles remain, with automatic GC enabled. This finding does not
identify the cause of earlier GPU timing variation. Snapshot capture adds timed
metadata work; its CPU cost must be recorded, without subtracting it from the
measurements or claiming zero overhead. Client and server still share a process,
so the result includes their normal GC and GIL interactions.

Freeze the reviewed source and operator bounds before one complete amended
18-process run. A separate baseline-readiness allocation is not required. Every
N/B and R/B gate and every correctness/lifecycle check must pass. If any fails,
retain all evidence and investigate before another method change; do not retry
unchanged, select favorable windows or pairs, waive N failures, or loosen gates.

The optional `stability` diagnostic keeps its original 200-request/12-second
windows and 90-second future wait through scoped settings that are restored on
success and failure. It records the current observer mode and worker source
hash. A new invocation uses this checkout's snapshot observer and is **not** a
byte-identical replay of the earlier diagnostic. Exact historical replay uses
commit `482b599224920a97bb2fce4a740317efb89351ab` and its original protocol.

## Evidence and CPU checks

`summary.json` schema 2 retains trial outcomes, aggregate metrics, statistical
gates and source hashes. Each entry links its complete `trial.json` by relative
path and SHA-256; raw request arrays and oracle corpora stay in those per-trial
files instead of being copied into the growing summary. Final cross-process
checks reopen the complete records and verify their hashes and identities.
The runner records its monotonic start/deadline and every admission timestamp,
full cleanup reserve and remaining margin, including a refused admission.
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
