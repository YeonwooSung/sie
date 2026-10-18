The new fixed three-arm L4 comparison completed all 18 trials. It keeps the sequential latency benefit and improves C8 burst fusion over the prior `sleep(0)` implementation, but it does not meet the issue's strict one-batch requirement for every HTTP burst. The result remains `completed_valid: true`, `qualification_passed: false`, with the existing human review hold unchanged.

The exact source arms were A: original timed coalescing `b0c26b02d21dbd146646c41b2b4f42627a0c9ebd`; B: current PR658 `8845eab5274ad571e0270c9d8c842035b5b5a426`; C: ready-callback candidate `a7c024f6f53cce0f8e43825845c5ad6965c9c792`. C preserves the PR history and changes only the lone-idle scheduling checkpoint plus its regression tests. A callback completes a Future so the waiting worker resumes after tasks created by already-ready producer callbacks. It is a finite scheduling opportunity; later network bodies or unfinished preprocessing can still miss it.

All arms used the same single NVIDIA L4, 8 CPU cores, 32 GiB memory, pinned `knowledgator/gliclass-large-v1.0` revision `ae8a18644e84cfda53325bd91c7941f2f6fd0ccb`, and actual local SIE HTTP API through `sie_sdk.SIEClient`. The observed model parameters were FP16 on CUDA. The default profile retained its existing bucketed CUDA graphs, threshold 0, batch-token limit 16384, and `idle_coalesce_ms=3`. Each fresh server used asyncio, four Torch intra-op threads and one inter-op thread. Python 3.12.12, Torch 2.9.1+cu129, Transformers 4.57.6, GLiClass 0.1.17, gliformer 0.1.2, httpx 0.28.1, and uvicorn 0.46.0 were admitted exactly. CPU scheduling/cancellation controls also cover uvloop; this GPU experiment did not use uvloop.

The six fixed triplets were ABC, CBA, BCA, ACB, CAB, BAC. Every trial warmed C1/C8/C32 with 64 calls each, measured 100 sequential calls, four idle bursts each at C8 and C32, and two 12-second closed-loop throughput phases, each preceded by 128 warmup calls. Throughput includes the full drain after the admission cutoff. All 119,432 SDK calls, including warmups, completed with one HTTP 200 response each. Full request/response bytes, SDK results, timing headers, and request-to-worker-batch joins were retained; no trial, request, pair, or outlier was excluded.

| Observation | A: timed coalescing | B: `sleep(0)` | C: ready callback |
|---|---:|---:|---:|
| Sequential queue samples below 1 ms | 0/600 | 600/600 | 600/600 |
| Per-trial queue median range | 3.395–3.440 ms | 0.090–0.100 ms | 0.100–0.110 ms |
| Per-trial queue nearest-rank p95 range | 3.84–4.27 ms | 0.11–0.13 ms | 0.13–0.17 ms |
| Maximum sequential queue time | 4.44 ms | 0.31 ms | 0.25 ms |
| Per-trial SDK latency p95 range | 14.469–15.779 ms | 10.840–11.373 ms | 10.987–12.363 ms |
| C8 bursts forming one worker batch | 24/24 | 3/24 | 18/24 |
| C32 bursts forming one worker batch | 0/24 | 0/24 | 0/24 |

C's six remaining C8 bursts split `[1, 7]`. At C32, the original baseline also split every burst; this does not waive the issue's one-batch criterion. The full ordered pattern inventory is below.

C/A throughput geometric ratios were 1.01095 at C8 (pointwise 95% interval 0.99637–1.02574) and 1.00010 at C32 (0.97286–1.02811). C/B ratios were 0.99319 (0.97069–1.01621) and 0.98980 (0.97423–1.00561). These are six paired fresh-process observations on one shared device/session. There is no predefined equivalence margin, so intervals containing 1 do not establish "unchanged within noise" or statistical equivalence. All twelve directions, individual ratios, and paired log dispersions are retained below.

All responses retained the same classification label sets, ordered label lists, winning labels, and other nonnumeric output structure. Sequential outputs matched the fixed first-A reference exactly. The largest direct score difference over all phases was 0.00390625 in each arm, and the same maximum range occurred within each arm's own repeats. Those differences remain visible; no output tolerance or task-quality claim is introduced.

The earlier six-trial yield result and its C8 split failure remain unchanged (`results.json` SHA256 `0c89a0b4693cffe29225415eace97e6551c2b426ca4eb99693b5da75c347ed4b`). The first attempt at this new three-arm protocol ended before inference because the cache verifier compared a lexical mount alias against a canonical root. Its failure was retained; run2 changed only that verified path-handling defect and attempt identities, after CPU alias/escape/retarget controls. No measured workload, output policy, trial order, or timing bound was changed in response to performance results.

The experiment includes identical output capture and batch tracing in every arm. It shares one device, image, and session across the balanced process order, so thermal, cache, JIT, and process variation remain possible. It measures the Python worker's HTTP path, not gateway/sidecar-only scheduling. The method does not predict arbitrary later network arrivals. The measured candidate has not yet been integrated with newer main; any later integration needs explicit source mapping and fresh CI/review.

The complete archive was verified before analysis. The provider was stopped and independently observed with zero containers and no tasks, every owned local process was reaped, and all experiment leases were released. No further GPU attempt is proposed.

Evidence SHA256 values: frozen method `fcf1aa6eac0542065c12f93c952d60b6b5d2aa7413455b2e3a97fa3c8437a8c2`; source-input manifest `398f68d9d62639a389fdb3b07850f19120a7eeda14d6edcaecf723e291147c35`; unchanged analyzer `b323ceedeb056f27bf40729ebd1749b8629a3b29bc4887ad866a83a55c859c08`; complete analysis `51e0d08bb724d08a5b270b3ca051ff917462c5b3cbed709553bbbba52a2c2c1f`; terminal receipt `cb99a4ee8e090a7a0fd7de8c0db41549b547bbce92fd1bc8ef332eb86b9d5998`; full 470-member archive `9933b24ff54bc85d6c617ba053aaf27a34ada8bc2ba69370f3d3dc70027fa9ed` (33,591,363 compressed bytes; 332,647,329 raw bytes). These identify retained artifacts; they are not a claim that private storage locations are publicly accessible. The complete method and result tables follow; the checked-in two-arm harness and its reproducibility limits are described at the end.

All 18 trials, in execution order (nearest-rank p95; block/trial numbers below start at zero).

| Trial | Block | Arm | Queue median / p95 (ms) | SDK p95 (ms) | C8 requests/s | C32 requests/s | C8 fused | C32 fused |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | A | 3.410 / 4.18 | 14.969 | 209.678 | 284.718 | 4/4 | 0/4 |
| 1 | 0 | B | 0.090 / 0.11 | 10.840 | 206.797 | 288.443 | 0/4 | 0/4 |
| 2 | 0 | C | 0.110 / 0.13 | 11.222 | 211.718 | 291.785 | 3/4 | 0/4 |
| 3 | 1 | C | 0.110 / 0.13 | 11.296 | 209.282 | 275.702 | 3/4 | 0/4 |
| 4 | 1 | B | 0.090 / 0.12 | 11.148 | 208.874 | 279.815 | 0/4 | 0/4 |
| 5 | 1 | A | 3.440 / 3.84 | 14.469 | 210.761 | 282.735 | 4/4 | 0/4 |
| 6 | 2 | B | 0.100 / 0.13 | 11.296 | 215.243 | 282.947 | 2/4 | 0/4 |
| 7 | 2 | C | 0.100 / 0.13 | 10.987 | 209.845 | 274.439 | 2/4 | 0/4 |
| 8 | 2 | A | 3.410 / 4.23 | 15.779 | 202.622 | 284.480 | 4/4 | 0/4 |
| 9 | 3 | A | 3.395 / 4.27 | 15.429 | 200.816 | 276.213 | 4/4 | 0/4 |
| 10 | 3 | C | 0.110 / 0.14 | 11.683 | 201.679 | 280.336 | 4/4 | 0/4 |
| 11 | 3 | B | 0.100 / 0.13 | 11.373 | 208.513 | 282.631 | 0/4 | 0/4 |
| 12 | 4 | C | 0.110 / 0.17 | 12.363 | 211.302 | 280.297 | 4/4 | 0/4 |
| 13 | 4 | A | 3.430 / 4.13 | 14.901 | 209.259 | 281.434 | 4/4 | 0/4 |
| 14 | 4 | B | 0.090 / 0.12 | 10.985 | 214.833 | 286.102 | 0/4 | 0/4 |
| 15 | 5 | B | 0.100 / 0.12 | 11.329 | 210.389 | 283.979 | 1/4 | 0/4 |
| 16 | 5 | A | 3.435 / 4.26 | 14.853 | 209.371 | 276.820 | 4/4 | 0/4 |
| 17 | 5 | C | 0.105 / 0.13 | 11.233 | 212.252 | 284.243 | 2/4 | 0/4 |

All ratio directions use six process pairs. Intervals are pointwise, two-sided 95% t intervals on paired log ratios (df = 5), without an equivalence margin.

| Throughput ratio | Six block ratios | Geometric mean | 95% interval | Paired log SD |
|---|---|---:|---|---:|
| throughput32:A/B | 0.98709, 1.01044, 1.00542, 0.97729, 0.98369, 0.97479 | 0.98969 | 0.97433–1.00530 | 0.014912 |
| throughput32:A/C | 0.97578, 1.02551, 1.03659, 0.98529, 1.00406, 0.97389 | 0.99990 | 0.97266–1.02790 | 0.026316 |
| throughput32:B/A | 1.01308, 0.98967, 0.99461, 1.02323, 1.01658, 1.02586 | 1.01041 | 0.99473–1.02635 | 0.014912 |
| throughput32:B/C | 0.98855, 1.01492, 1.03100, 1.00818, 1.02071, 0.99907 | 1.01031 | 0.99442–1.02645 | 0.015102 |
| throughput32:C/A | 1.02482, 0.97513, 0.96470, 1.01493, 0.99596, 1.02681 | 1.00010 | 0.97286–1.02811 | 0.026316 |
| throughput32:C/B | 1.01159, 0.98530, 0.96993, 0.99188, 0.97971, 1.00093 | 0.98980 | 0.97423–1.00561 | 0.015102 |
| throughput8:A/B | 1.01393, 1.00904, 0.94136, 0.96309, 0.97406, 0.99516 | 0.98243 | 0.95311–1.01266 | 0.028872 |
| throughput8:A/C | 0.99036, 1.00707, 0.96558, 0.99572, 0.99033, 0.98643 | 0.98917 | 0.97490–1.00365 | 0.013844 |
| throughput8:B/A | 0.98626, 0.99104, 1.06229, 1.03833, 1.02664, 1.00486 | 1.01788 | 0.98750–1.04920 | 0.028872 |
| throughput8:B/C | 0.97676, 0.99805, 1.02572, 1.03389, 1.01671, 0.99122 | 1.00686 | 0.98405–1.03020 | 0.021838 |
| throughput8:C/A | 1.00973, 0.99298, 1.03565, 1.00430, 1.00976, 1.01376 | 1.01095 | 0.99637–1.02574 | 0.013844 |
| throughput8:C/B | 1.02380, 1.00196, 0.97492, 0.96722, 0.98357, 1.00885 | 0.99319 | 0.97069–1.01621 | 0.021838 |

Every measured idle-burst pattern (ordered worker batch sizes; 24 bursts per arm and concurrency).

| Arm | Concurrency | Pattern counts |
|:---:|---:|---|
| A | 8 | `[8]` × 24 |
| A | 32 | `[19, 13]` × 1; `[21, 11]` × 1; `[25, 7]` × 3; `[26, 6]` × 6; `[27, 5]` × 6; `[28, 4]` × 4; `[29, 3]` × 2; `[30, 2]` × 1 |
| B | 8 | `[1, 7]` × 21; `[8]` × 3 |
| B | 32 | `[1, 15, 16]` × 2; `[1, 31]` × 15; `[25, 7]` × 1; `[27, 5]` × 3; `[28, 4]` × 3 |
| C | 8 | `[1, 7]` × 6; `[8]` × 18 |
| C | 32 | `[1, 11, 20]` × 1; `[1, 12, 19]` × 1; `[1, 23, 8]` × 1; `[1, 31]` × 2; `[22, 10]` × 1; `[24, 8]` × 1; `[26, 6]` × 6; `[27, 5]` × 4; `[28, 4]` × 6; `[31, 1]` × 1 |

The public serving source and original two-arm workload are in this repository. The existing `tools/benchmarks/idle_worker/README.md` documents source preparation and execution on a caller-owned GPU. Once the measured candidate commit is published, a fresh original-versus-candidate comparison can use:

```sh
mise exec -- python tools/benchmarks/idle_worker/prepare.py \
  --baseline b0c26b02d21dbd146646c41b2b4f42627a0c9ebd \
  --candidate a7c024f6f53cce0f8e43825845c5ad6965c9c792
mise exec -- python tools/benchmarks/idle_worker/run.py \
  --output /tmp/sie-idle-worker-results
```

These commands run the existing two-arm benchmark, not the completed three-arm experiment verbatim. The 18-trial experiment retained additional full request/response bytes, source/runtime/cache admission, exact process ownership, failure persistence, and a separate full-row analyzer. Those additions are not yet packaged as a public one-command reproducer. Its fixed method is specified above so that the difference is visible; do not treat a new run of the older harness as a hash-identical replay or pool it with this result.

For a faithful independent implementation of the three-arm method, preserve all six listed triplets on the same device, fresh actual server/SDK instances, the exact environment/profile and fixed public checkpoint, all warmup/count/duration/drain rules, 50 ms before each idle burst, and all original raw output/batch observations. The full source archive path inventory consisted of server source, SDK source, model catalog, bundle definitions, and `uv.lock`; the same lock SHA256 was `c1286da62744012fe54f07cd7ec740d1d1a69db0e2db75c8a0a0a8007ac3a397`. The A/B/C archive SHA256 values were `0926ef65a420c437f9f5f2147a44f0b6ed07e0cd51cf7a376a2bb99eccee2257`, `6e8ba5a26bd3c2f223cd2bb05351e53307c0180d645a4541678a9fb8fded29d0`, and `e2380107510882bf8c273d6139e79d63412a31327fe4365a45f335656512928f`.

The fixed active-run ceiling was 1800 seconds, with 180 seconds active plus at most 30 seconds cleanup for each admitted fresh trial. The experiment stopped on the first failure, required enough remaining budget for a complete trial and cleanup, and did not append blocks, shorten windows, or omit outliers. Independent process/phase identity and every complete request/wire/batch join are necessary to reproduce the integrity assessment; timing similarity alone is insufficient.

The seven required public snapshot files totaled 1,763,619,203 bytes. The checkpoint `model.safetensors` was 1,752,499,444 bytes with SHA256 `9d6bafde8a1a43bf5563fd6296f8934752d23462de9113b0b8ac56ecc7cdddec`; the required-file manifest SHA256 was `c003d0e73df9bb7d23337c6d6a54312dc07e2639704e47981b15b158812ab053`. Full file content, revision, and resolved cache paths were checked before model work. A cache or environment failure is an incomplete attempt, not a reason to substitute inputs silently.

Independent reviewers recomputed all 119,432 request joins, 288 phases, 19,221 post-forward worker batches, every ratio and output diagnostic. Worker batch records identify SIE dispatch grouping; they do not establish physical CUDA tensor shapes or the cause of a particular HTTP split. The result-review SHA256 is `9a00f78392a47f60f5eff6956fa0037dc9468d1d31e19efdf2109905c2c28b08`; the independent terminal/archive review SHA256 is `725bbd0d6f842ccceb4416175dcb3718af4998182e03c2dde496290306a5d8b2`.
