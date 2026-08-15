# inferno

A mock LLM inference engine

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r scripts/requirements.txt

uvicorn test:app --reload       # starts the mock server on :8000
```

In another terminal:

```bash
curl -X POST localhost:8000/generate \
  -H "content-type: application/json" -d '{"prompt_id": 0}'
```

```json
{
  "prompt_id": 0,
  "model": "sim-8b-instruct",
  "content": "[simulated completion | prompt_id=0] professional option value ...",
  "prompt_tokens": 10,
  "completion_tokens": 79,
  "total_tokens": 89,
  "timing_ms": { "queue": 0.05, "ttft": 12.68, "total": 1329.18, "per_output_token": 16.66 }
}
```

`prompt_id` is the row index of a prompt in the bundled ShareGPT dataset
(`0` to `len(PROMPTS) - 1`, currently up to 999). The completion is filler
text assembled from a **word bank built out of the dataset's own
vocabulary**, sized so that re-counting the returned text gives back exactly
the `completion_tokens` the API reported — the numbers are internally
consistent even though nothing was actually generated.

### Helper scripts

Instead of typing curl by hand:

```bash
python scripts/loadtest.py          # 5 sequential POST /generate, pretty-printed
python scripts/loadtest.py --count 20
python scripts/show_metrics.py      # pretty-printed GET /metrics
```

---

## Endpoints

| endpoint | |
|---|---|
| `POST /generate` | `{"prompt_id": int}` → completion + per-stage timings |
| `GET /metrics` |  
| `GET /health` | `{"status", "records", "vocab"}` | curl -s localhost:8000/health

### `/metrics`

Returns plain JSON (not Prometheus exposition format — this is meant to be
read by a person, not scraped):

```json
{
  "requests_running": 0,
  "requests_waiting": 0,
  "kv_cache_usage_percent": 0.0, 
  "prompt_tokens_total": 1373,
  "generation_tokens_total": 226,
  "latency_seconds": {
    "time_to_first_token": { "count": 2, "avg": 0.09, "p50": 0.17, "p90": 0.17, "p99": 0.17, "max": 0.17 },
    "time_per_output_token": { "...": "..." },
    "inter_token_latency": { "...": "..." },
    "e2e_request_latency": { "...": "..." },
    "request_queue_time": { "...": "..." },
    "request_prefill_time": { "...": "..." },
    "request_decode_time": { "...": "..." },
    "request_inference_time": { "...": "..." }
  }
}
```

Every latency metric reports `count`, `avg`, `min`, `p50`, `p90`, `p99`, `max`
in seconds, computed from every request served since the process started.

| metric | meaning |
|---|---|
| `requests_running` | requests currently occupying a batch slot (in prefill or decode right now) |
| `requests_waiting` | requests queued, waiting for a free batch slot |
| `kv_cache_usage_percent` | how full the KV cache is; watch this to see if the KV cache — not the batch slots — is the bottleneck |
| `prompt_tokens_total` | cumulative prompt (prefill) tokens processed since startup |
| `generation_tokens_total` | cumulative completion (decode) tokens produced since startup |
| `time_to_first_token` (TTFT) | arrival → first generated token, i.e. queue + tokenize + prefill |
| `time_per_output_token` (TPOT) | average time per generated token for a request: `(e2e - ttft) / completion_tokens` |
| `inter_token_latency` (ITL) | the actual duration of each individual decode step (one sample per token, not per request) |
| `e2e_request_latency` | arrival → final token; the full request lifetime |
| `request_queue_time` | time spent in stage 0, waiting for a batch slot |
| `request_prefill_time` | time spent in stage 2 alone (the initial forward pass over the prompt) |
| `request_decode_time` | time spent in stage 3 alone (the autoregressive generation loop) |
| `request_inference_time` | total GPU-active time: prefill + decode |

---

## The six stages

Every request walks through the same pipeline, timed with `time.perf_counter`
and `asyncio.sleep` so the delays are real:

```
arrival
   │
   ▼
stage 0  admission/queue  —    waiting for a free batch slot (MAX_BATCH_SIZE)
   │
   ▼
stage 1  tokenization     CPU   text -> token IDs (approximated by count only)
   │
   ▼
stage 2  prefill          GPU  one pass over the whole prompt, compute-bound
   │                            → this is what determines TTFT
   ▼
stage 3  decode           GPU  autoregressive loop, one token/iteration,
   │                            memory-bandwidth-bound → cost/token is ~constant
   ▼
stage 4  detokenization   CPU  token IDs -> text
   │
   ▼
stage 5  teardown         —    free KV blocks, admit the next queued request
```



Six constants at the top of [`test.py`](test.py):

```python
PREFILL_MS_PER_TOKEN   = 0.10    # compute-bound: 2 * params * tokens FLOPs
DECODE_MS_PER_TOKEN    = 12.0    # bandwidth-bound: model_bytes / HBM bandwidth
TOKENIZE_MS_PER_KCHAR  = 0.20    # CPU, Rust tokenizer throughput
DETOKENIZE_MS_PER_TOK  = 0.02    # CPU, negligible
MAX_BATCH_SIZE         = 16      # continuous-batching slot count
KV_CACHE_TOKENS        = 100_000 # total KV-cache capacity, in tokens
```

Tuned so the simulator behaves roughly like an ~8B model on one A100. Change
them and re-run to see how the timings and queueing behavior shift.

---

## Layout

```
test.py                       the mock server -- dataset load, tokenizer approximation,
                               fake completions, metrics, the six stages, FastAPI routes
data/
  sharegpt-prompts-1k.csv     the bundled prompt dataset
scripts/
  loadtest.py                 fires N POST /generate requests, prints curl + response
  show_metrics.py             fetches and pretty-prints GET /metrics
  estimate_tokens.py          benchmarks cheap token-count approximations vs tiktoken
  download_sharegpt_prompts.py  regenerates data/sharegpt-prompts-1k.csv (needs Kaggle creds)
  requirements.txt
```

To regenerate the dataset (needs Kaggle API credentials — `kaggle.json` in
`~/.kaggle/`, or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars):

```bash
python scripts/download_sharegpt_prompts.py
```

