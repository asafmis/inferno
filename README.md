# inferno

**A mock LLM inference engine, built to understand how real ones (vLLM, TGI, TensorRT-LLM) work.**

This is an educational, experimental project. It does not run a model — there
are no weights, no CUDA, no attention kernels. Instead it *simulates* a serving
engine: a virtual clock, a scheduler, a paged KV cache, and a cost model that
says how long each operation would take on real hardware.

The point is that almost everything surprising about LLM serving is a
**systems** property, not a model property. Why does p99 latency explode while
the median looks fine? Why does batching give 60× throughput? Why does a
request sit idle for eight seconds before a single FLOP is spent on it? None of
those questions need a GPU to answer — they need a queue, a cache, and an
honest cost model. That is what this repo is.

There are two ways in, and they answer different questions:

| | [`app.py`](app.py) | [`inferno/`](inferno) + [`scripts/run_simulation.py`](scripts/run_simulation.py) |
|---|---|---|
| what it is | a **running HTTP service** that mocks vLLM | an **offline simulator** of the engine |
| time | real — it actually sleeps | virtual — 5 minutes of traffic in 0.2 s |
| use it to | develop clients, dashboards, autoscalers against a fake vLLM | explore scheduler behaviour, sweep parameters, read the code |
| size | one file, ~370 lines | 8 modules |

```bash
python app.py                                       # the mock server, port 8000
python scripts/run_simulation.py --explain          # per-stage cost walkthrough
python scripts/run_simulation.py --sweep 1,2,4,8,16,32
```

---

## The mock server

One file, [`app.py`](app.py). No model, no GPU, no weights — it sleeps for the
time a real engine would have spent and exposes the metrics vLLM exposes.

```bash
pip install -r scripts/requirements.txt
python app.py
```

```bash
curl -X POST localhost:8000/generate \
  -H "content-type: application/json" -d '{"prompt_id": 140731}'
```

```json
{
  "prompt_id": 140731,
  "content": "[simulated completion | prompt_id=140731] ratio clients short idea knows ...",
  "prompt_tokens": 10,
  "completion_tokens": 150,
  "total_tokens": 160,
  "timing_ms": { "queue": 4.86, "ttft": 6.75, "total": 2525.55, "per_output_token": 13.31 }
}
```

`prompt_id` refers to a prompt in the ShareGPT dataset (`GET /prompts` lists
them). The completion is filler assembled from a **word bank built out of the
dataset's own vocabulary**, sized so that re-counting the returned text gives
back exactly the `completion_tokens` the API reported.

| endpoint | |
|---|---|
| `POST /generate` | `{"prompt_id": int, "seed": int?}` → completion + per-stage timings |
| `GET /metrics` | Prometheus exposition, vLLM metric names |
| `GET /prompts?limit=N` | discover valid `prompt_id`s |
| `GET /health` | live `running` / `waiting` / `kv_usage` |

The knobs are six constants at the top of the file:

```python
PREFILL_MS_PER_TOKEN   = 0.10   # compute-bound: 2 * params * tokens FLOPs
DECODE_MS_PER_TOKEN    = 12.0   # bandwidth-bound: model_bytes / HBM bandwidth
TOKENIZE_MS_PER_KCHAR  = 0.20   # CPU, Rust tokenizer throughput
DETOKENIZE_MS_PER_TOK  = 0.02   # CPU, negligible
MAX_BATCH_SIZE         = 16     # continuous-batching slot count
KV_CACHE_TOKENS        = 32_768 # total KV capacity, in token positions
```

### Metrics

Same names and semantics as vLLM's, so an existing dashboard or alert rule
works unchanged:

```
vllm:num_requests_running          gauge      requests on the "GPU"
vllm:num_requests_waiting          gauge      requests queued in stage 0
vllm:kv_cache_usage_perc           gauge      fraction of KV capacity in use
vllm:time_to_first_token_seconds   histogram  TTFT
vllm:e2e_request_latency_seconds   histogram  end-to-end latency
vllm:prompt_tokens_total           counter    prefill tokens processed
vllm:generation_tokens_total       counter    generation tokens produced
```

### Watching stage 0 appear

```bash
python scripts/loadtest.py 48     # 48 concurrent requests, 16 slots
```

```
wall clock        23.39 s
throughput        2.05 req/s   |  513 output tok/s

                 p50       p90       p99       max   ms
queue         3528.8    7935.2    8280.5    8280.5
ttft          3575.0    7937.6    8314.8    8314.8
e2e           7184.6   14310.6   22455.4   22455.4

queue was 43% of mean end-to-end latency
```

With 48 requests against 16 slots, the median request waits **3.5 seconds
before a single simulated FLOP is spent on it**, and queueing accounts for 43%
of end-to-end latency. Nothing about the "model" changed — this is stage 0
alone.

> **Two footguns this shook out**, both worth knowing if you build something
> similar. First, the batch-slot semaphore has to be held for the request's
> *entire* life; releasing it once the KV cache is reserved lets everything
> decode at once and no queue ever forms. Second, on Windows the default timer
> granularity is ~15.6 ms, so `asyncio.sleep(0.012)` actually takes ~17 ms and
> every per-token number comes out 40% high — `app.py` requests a 1 ms timer at
> startup, which brings it to ~12.7 ms.

---

## The six stages

Every request walks through the same pipeline. The simulator models each one
explicitly, and the final report attributes latency back to them.

```
   arrival
      │
      ▼
 ┌─ stage 1 ──────────┐   tokenization       CPU    text → token IDs
 │  ~0.5 ms            │                             runs on a worker pool
 └────────┬────────────┘
          ▼
 ┌─ stage 0 ──────────┐   admission/queue    —      waiting for a batch slot
 │  0 … unbounded      │                             and for free KV blocks
 └────────┬────────────┘
          ▼
 ┌─ stage 2 ──────────┐   prefill            GPU    one pass over the whole
 │  ~105 ms / 1k tok   │   compute-bound             prompt → emits token 1
 └────────┬────────────┘
          ▼
 ┌─ stage 3 ──────────┐   decode             GPU    autoregressive loop,
 │  ~9 ms / token      │   bandwidth-bound           one token per iteration
 └────────┬────────────┘
          ▼
 ┌─ stage 4 ──────────┐   detokenization     CPU    token IDs → text,
 │  ~10 µs / token     │                             incremental in streaming
 └────────┬────────────┘
          ▼
 ┌─ stage 5 ──────────┐   teardown           —      free KV blocks; this is
 │  ~0 ms              │                             what admits the next request
 └─────────────────────┘
```

### Stage 0 — admission and queueing

The part people forget, and the difference between a good median and a
catastrophic p99. A request lands in a FIFO queue and the scheduler admits it
only when there is both a batch slot and enough free KV cache. On an idle
engine this costs nothing. On a saturated one it dominates every other stage
*combined* — see the load sweep below, where queueing goes from 4 ms to 388 ms
while nothing about the model changed.

Implemented in [`inferno/scheduler.py`](inferno/scheduler.py).

### Stage 1 — tokenization

Text → integer IDs. Pure CPU, no GPU involvement. A Rust `tokenizers`
implementation does roughly 1–10 MB/s per core, so a 2,000-character prompt
costs somewhere around 0.2–2 ms — effectively free per request. It only matters
in aggregate: at thousands of requests per second it competes with the process
driving the GPU, which is why vLLM runs a dedicated tokenizer pool. The
simulator models that pool (`--tokenizer-workers`) mostly to demonstrate that
it is not the bottleneck.

### Stage 2 — prefill

One forward pass over the entire prompt. Every position is processed in
parallel, so the GPU is doing large dense matmuls and is **compute-bound**. This
populates the KV cache for the prompt and emits exactly one token — the first.

```
cost ≈ 2 × params × prompt_tokens FLOPs
```

For 8B params that is ~16 GFLOP per prompt token; at ~156 TFLOPS effective, a
1,000-token prompt takes ~105 ms. There is also a quadratic attention term,
which is 1.6% of the total at 1k tokens and >10% by 8k — negligible until it
suddenly isn't. **Cost scales with prompt length.**

### Stage 3 — decode

The autoregressive loop. Each iteration embeds the previous token, runs all
layers attending over the cached KV, projects to logits, samples, and appends
the new KV. Repeat until EOS.

Each step touches *one* token position but must read *every weight* from HBM,
so it is **memory-bandwidth-bound**:

```
cost ≈ model_bytes / memory_bandwidth
```

8B at bf16 is 16 GB over ~1.84 TB/s effective ≈ 8.8 ms per token at batch 1.
**Cost per token is roughly constant in prompt length** — that is precisely
what the KV cache buys you. The prompt is never re-read.

### Stage 4 — detokenization

Token IDs → text, on the CPU, microseconds per token. In streaming mode it runs
once per generated token. It is *off* the GPU critical path: it delays when the
client sees a token without delaying the next forward pass, which is how the
simulator models it.

### Stage 5 — teardown

Free the KV blocks. Costs no meaningful time, but it belongs in the diagram
because it is the event that unblocks stage 0 for the next queued request.

---

## The one equation

Stages 2 and 3 are usually described as two different things. They are not —
they are the same forward pass under a **roofline**:

```python
iteration_time = max(flops / compute_throughput, bytes_moved / bandwidth)
```

* **Prefill** puts thousands of token positions through the numerator of the
  FLOPs term while the bytes term is just "read the weights once". Compute
  wins → compute-bound.
* **Decode** puts one token position per sequence through it. The FLOPs term
  collapses; the weights still have to be read in full. Memory wins →
  memory-bound.

And continuous batching falls out for free: the weight-read cost is paid once
per **iteration**, not once per **sequence**. Adding sequences to a decode batch
is nearly free until the batch grows big enough to tip back over the roofline:

```
batch    1     9.32 ms/step        107 tok/s   [memory-bound]
batch    8     9.82 ms/step        814 tok/s   [memory-bound]
batch   32    11.54 ms/step      2,773 tok/s   [memory-bound]
batch  128    18.41 ms/step      6,952 tok/s   [memory-bound]
```

128× the batch for 2× the step time — **63× the throughput**. This is the entire
economic argument for a batching inference server, and it is one `max()`.

---

## What the simulator shows you

### Latency is fine until it isn't

`python scripts/run_simulation.py --requests 400 --sweep 1,2,4,8,16,32`

```
  rate   goodput    tok/s    gpu  queue p50  ttft p50  ttft p99  itl p50  itl p99  kv pk  preempt
     1      0.98      254    88%      0.004     0.016     0.453      9.3      9.9     2%        0
     2      1.92      496    98%      0.005     0.016     0.460      9.4      9.9     2%        0
     4      3.55      918    99%      0.005     0.017     0.462      9.6     13.1     3%        0
     8      6.17    1,596   100%      0.007     0.019     0.698     10.0     67.9     5%        0
    16      9.70    2,512   100%      0.009     0.042     0.848     11.3    200.8    10%        0
    32     11.54    2,988   100%      0.388     0.837     4.088     25.5    460.3    24%        0
```

Read the `ttft p50` and `ttft p99` columns together. At 16 req/s the median user
sees 42 ms and thinks the service is fast; the unlucky 1% sees 848 ms — 20×
worse. The median gives no warning at all before the cliff at 32 req/s. Note
also that KV cache peaks at 24% — this system saturates on **bandwidth**, not
memory, with these prompt lengths.

### The recompute death spiral

Starve the KV cache and watch what preemption actually costs:

```bash
python scripts/run_simulation.py --requests 300 --rate 24 --output-median 1500 \
  --output-cap 6000 --max-model-len 16384 --gpu-memory-utilization 0.30
```

```
  tokens per iteration 74.7 (prefill 716,468 / decode 709,736)
  preemptions          4,106
  kv cache             ... peak use 100.0%

  queue (stage 0)        82.39    211.10    211.49    211.57     76.79   s
```

**716,468 prefill tokens computed for 81,018 tokens of actual prompt** — a 9×
amplification. vLLM's default preemption policy is *recompute*: rather than
swapping KV to host memory, it drops the blocks and re-prefills from scratch on
readmission. Cheap for one request, ruinous as a feedback loop, because the
recompute work makes the cache pressure worse. Stage 0 becomes 54% of
end-to-end latency.

### Chunked prefill

```bash
python scripts/run_simulation.py --requests 300 --rate 12                       # ITL p90 11.0 ms, max  930 ms
python scripts/run_simulation.py --requests 300 --rate 12 --no-chunked-prefill  # ITL p90 19.4 ms, max 1529 ms
```

With chunked prefill off (the older vLLM policy), an iteration runs prefills
*or* decodes, never both — so one long prompt stalls every decoding request
behind it, and everyone's inter-token latency spikes. Chunked prefill splits the
prompt across iterations so each batch stays roughly the same size.

---

## Layout

```
app.py          the mock vLLM server -- one file, six stages, Prometheus metrics
inferno/
  config.py     ModelConfig / GPUConfig / EngineConfig, plus presets and tensor parallelism
  request.py    the Request state machine and its stage timestamps
  costs.py      the roofline cost model — the only file where physics happens
  kv_cache.py   paged block allocator (stage 5's counterpart)
  scheduler.py  continuous batching: admission, chunked prefill, preemption (stage 0)
  engine.py     the discrete-event loop that wires all six stages together
  metrics.py    TTFT / ITL / E2E percentiles and the stage breakdown
  workload.py   arrival processes and prompt/output length distributions
scripts/
  run_simulation.py    CLI: single run, --explain walkthrough, --sweep load curve
  verify_simulator.py  31 invariant and physics checks
  loadtest.py          concurrent load against app.py, to watch the queue form
  estimate_tokens.py   benchmark of cheap token-count approximations vs tiktoken
  download_sharegpt_prompts.py
data/
  sharegpt-prompts-1k.csv
```

There is no real-time sleeping anywhere: simulating five minutes of traffic
takes a fraction of a second, so you can sweep a parameter across a dozen values
in the time it takes to read the output.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r scripts/requirements.txt
```

The ShareGPT prompt dataset is committed under `data/`. To regenerate it you
need Kaggle API credentials:

```bash
python scripts/download_sharegpt_prompts.py
```

---

## Useful flags

| Flag | What it explores |
|---|---|
| `--rate`, `--arrival {poisson,uniform,burst}` | How much of your queueing is caused by burstiness rather than raw load |
| `--sweep 1,2,4,8,16` | The load curve, and where the p50/p99 gap opens up |
| `--no-chunked-prefill` | The TTFT-vs-ITL tradeoff |
| `--gpu-memory-utilization` | Shrink the KV cache until preemption starts |
| `--max-num-batched-tokens` | Caps iteration length, which caps everyone's ITL |
| `--output-median`, `--output-sigma` | Long-tail responses and the KV pressure they create |
| `--model llama3-70b --tensor-parallel 4` | Sharding a model too big for one card |
| `--length-source estimate` | Admitting requests on an *estimated* token count instead of a real one |

That last one connects to `scripts/estimate_tokens.py`, which benchmarks six
cheap "how many tokens is this?" formulas against tiktoken on the ShareGPT
prompts. Counting word-and-punctuation atoms and scaling by 1.1 wins at ~9%
MAPE, roughly half the error of the familiar chars/4 rule:

```
formula                    MAPE   median      max
1 chars/4                 22.0%    19.6%   100.0%
2 words*1.33              17.3%    14.3%    92.7%
3 atoms*1.1                9.3%     7.7%    72.7%
4 subword-aware           23.2%    21.9%    85.7%
5 hybrid floor            21.9%    19.6%   100.0%
6 char-class weighted     21.8%    19.6%   100.0%
```

---

## How much should you trust the numbers?

First-order, not kernel-accurate. The cost model uses spec-sheet FLOPs and
bandwidth scaled by efficiency factors (50% and 90% by default), which
reproduces the standard back-of-envelope figures:

| quantity | whiteboard | simulator |
|---|---|---|
| prefill, 1k tokens, 8B | ~100 ms | 105 ms |
| decode, batch 1, 8B | ~8 ms/token | 8.8 ms/token |
| KV cache, Llama-3-8B | 128 KiB/token | 128 KiB/token |

`scripts/verify_simulator.py` checks 31 invariants: conservation (every KV block
returned, every token accounted for), causality (timestamps only move forward),
agreement with the closed-form arithmetic, and Little's Law (`L = λW`) as an
independent consistency check between the scheduler's behaviour and the
per-request timestamps.

```bash
python scripts/verify_simulator.py
```

**What is deliberately not modelled:** actual model quality or sampling, prefix
caching, speculative decoding, quantization, pipeline parallelism, LoRA
adapters, multi-node networking, CPU-GPU transfer for swap-based preemption, and
kernel-level effects like wave quantization. Output lengths are drawn from a
lognormal up front rather than discovered at EOS — the simulator gets to cheat
because it has no model to sample from.
