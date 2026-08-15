"""
LLM Inference Simulator -- a single-file mock of a vLLM-style serving engine.
 
No model is loaded and no GPU is used. The service reproduces the *shape* of
LLM inference: the stages a request passes through, how long each one takes,
and the metrics a real engine reports.
 
Run:   uvicorn app:app --reload
Call:  curl -X POST localhost:8000/generate -H 'Content-Type: application/json' \
            -d '{"prompt_id": 123}'
"""
 
import asyncio
import csv
import math
import os
import random
import re
import time
from collections import Counter
 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
 
# ---------------------------------------------------------------------------
# CONFIG -- tuned so the simulator behaves like an ~8B model on one A100.
# ---------------------------------------------------------------------------
PREFILL_MS_PER_TOKEN = 0.10    # compute-bound:   2 * params * tokens FLOPs
DECODE_MS_PER_TOKEN = 12.0     # bandwidth-bound: model_bytes / HBM bandwidth
TOKENIZE_MS_PER_KCHAR = 0.20   # CPU, Rust tokenizer throughput
DETOKENIZE_MS_PER_TOK = 0.02   # CPU, negligible
MAX_BATCH_SIZE = 16            # continuous-batching slot count
KV_CACHE_TOKENS = 100_000      # total KV-cache capacity, in tokens
DATASET_PATH = os.getenv("DATASET_PATH", r"data\sharegpt-prompts-1k.csv")
DATASET_COLUMN = os.getenv("DATASET_COLUMN", "original_prompt")
 
# ---------------------------------------------------------------------------
# 1. DATASET -- loaded into memory once at startup, never touched again.
#    Reading from disk inside the request path would pollute the timings we
#    are trying to simulate.
# ---------------------------------------------------------------------------
def load_dataset(path: str, column: str, limit: int = 1000) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = column if column in (reader.fieldnames or []) else (reader.fieldnames or [""])[-1]
        rows = [r[col].strip() for r in reader if r.get(col) and r[col].strip()]
    return rows[:limit]
 
 
PROMPTS = load_dataset(DATASET_PATH, DATASET_COLUMN)
 
# Vocabulary for the fake completions, harvested from the dataset itself so the
# generated text has the same word-length profile as the real prompts.
WORD_BANK = [
    w for w, _ in Counter(
        w.lower() for p in PROMPTS for w in re.findall(r"[A-Za-z]{2,12}", p)
    ).most_common(300)
] or ["the", "and", "for", "with", "that", "this", "from", "have"]
 
# ---------------------------------------------------------------------------
# 2. TOKENIZER APPROXIMATION
#    A real BPE tokenizer would return integer IDs; the simulator only needs
#    the *count*. We split into word-and-punctuation "atoms" -- which is close
#    to what BPE emits, since common words are one token and each punctuation
#    mark is its own -- then apply a flat 1.1x correction for rare words that
#    split into multiple pieces.
# ---------------------------------------------------------------------------
_ATOM_RE = re.compile(r"\w+|[^\w\s]")
 
 
def estimate_tokens(text: str) -> int:
    atoms = _ATOM_RE.findall(text)
    return max(1, math.ceil(len(atoms) * 1.1))
 
 
# ---------------------------------------------------------------------------
# 3. FAKE OUTPUT
# ---------------------------------------------------------------------------
def sample_output_length(rng: random.Random) -> int:
    """Real completion lengths are lognormal-ish: mostly short, long tail."""
    return max(1, min(1024, int(rng.lognormvariate(4.5, 0.8))))
 
 
def synthesize_completion(target_tokens: int, prompt_id: int, rng: random.Random) -> str:
    """Filler whose approximate token count equals target_tokens.
 
    Content is meaningless by design -- no model runs. What matters is that
    estimate_tokens(result) == the completion_tokens we report, so the
    response payload is internally consistent.
    """
    header = f"[simulated completion | prompt_id={prompt_id}] "
    words = [header.strip()]
    while estimate_tokens(" ".join(words)) < target_tokens:
        words.append(rng.choice(WORD_BANK))
    text = " ".join(words)
    while estimate_tokens(text) > target_tokens and len(text) > len(header):
        text = text[: text.rfind(" ")]
    return text.rstrip() + "."
 
 
# ---------------------------------------------------------------------------
# 4. METRICS -- the same things vLLM tracks (a per-request timing histogram
#    for each stage, live gauges, cumulative counters), but /metrics reports
#    plain readable summary stats. There is no scraper here, just a human
#    reading the response, so we skip Prometheus's text exposition format
#    entirely.
# ---------------------------------------------------------------------------
class Histogram:
    """Records raw observations, in seconds, and reports summary stats."""

    def __init__(self, description: str):
        self.description = description
        self._values: list[float] = []

    def observe(self, seconds: float) -> None:
        self._values.append(seconds)

    def summary(self) -> dict:
        if not self._values:
            return {"description": self.description, "count": 0}
        values = sorted(self._values)
        n = len(values)

        def percentile(p: float) -> float:
            return round(values[min(n - 1, int(p * n))], 4)

        return {
            "description": self.description,
            "count": n,
            "avg": round(sum(values) / n, 4),
            "min": round(values[0], 4),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p99": percentile(0.99),
            "max": round(values[-1], 4),
        }


ttft = Histogram("time to first token: arrival -> first generated token")
tpot = Histogram("time per output token: (e2e - ttft) / completion_tokens")
itl = Histogram("inter-token latency: duration of each individual decode step")
e2e = Histogram("end-to-end latency: arrival -> final token")
queue_time = Histogram("time spent WAITING for a batch slot")
prefill_time = Histogram("time spent in PREFILL")
decode_time = Histogram("time spent in DECODE")
inference_time = Histogram("time spent RUNNING: prefill + decode")

_state = {
    "requests_running": 0,
    "requests_waiting": 0,
    "kv_cache_usage_percent": 0.0,
    "prompt_tokens_total": 0,
    "generation_tokens_total": 0,
}
 
# ---------------------------------------------------------------------------
# 5. THE SIMULATION
# ---------------------------------------------------------------------------
_slots = asyncio.Semaphore(MAX_BATCH_SIZE)   # continuous-batching slots
_kv_used = 0                                 # tokens currently resident in KV cache
 
 
async def simulate_inference(prompt: str, prompt_id: int) -> dict:
    global _kv_used
    t0 = time.perf_counter()
    rng = random.Random(prompt_id)           # same id -> same output, so it is testable
 
    # ---- STAGE 0: ADMISSION ----------------------------------------------
    # Real engines hold requests in a waiting queue until a batch slot and
    # enough KV-cache blocks are free. The semaphore reproduces the queueing
    # delay that dominates tail latency under load.
    _state["requests_waiting"] += 1
    async with _slots:
        _state["requests_waiting"] -= 1
        _state["requests_running"] += 1
        queue_ms = (time.perf_counter() - t0) * 1000

        # ---- STAGE 1: TOKENIZATION ---------------------------------------
        # CPU-side, text -> token IDs. We approximate the count only; the
        # tokenizer's output length is all the timing model needs.
        prompt_tokens = estimate_tokens(prompt)
        await asyncio.sleep(len(prompt) / 1000 * TOKENIZE_MS_PER_KCHAR / 1000)

        # Materialise the output now so the token count that drives the decode
        # timing is exactly the count we report back to the caller.
        text = synthesize_completion(sample_output_length(rng), prompt_id, rng)
        completion_tokens = estimate_tokens(text)

        _kv_used += prompt_tokens + completion_tokens
        _state["kv_cache_usage_percent"] = round(min(1.0, _kv_used / KV_CACHE_TOKENS) * 100, 2)

        # ---- STAGE 2: PREFILL --------------------------------------------
        # One parallel forward pass over the whole prompt. Compute-bound, so
        # cost scales LINEARLY WITH PROMPT LENGTH. Fills the KV cache and
        # emits the first token -- this is what determines TTFT.
        prefill_t0 = time.perf_counter()
        await asyncio.sleep(prompt_tokens * PREFILL_MS_PER_TOKEN / 1000)
        prefill_ms = (time.perf_counter() - prefill_t0) * 1000
        ttft_ms = (time.perf_counter() - t0) * 1000

        # ---- STAGE 3: DECODE ---------------------------------------------
        # Autoregressive loop, one token per iteration. Memory-bandwidth-bound:
        # every step re-reads all model weights, so per-token cost is roughly
        # CONSTANT and INDEPENDENT OF PROMPT LENGTH -- that is what the KV
        # cache buys us. Each step's own duration is the inter-token latency.
        decode_t0 = time.perf_counter()
        for _ in range(completion_tokens):
            step_t0 = time.perf_counter()
            await asyncio.sleep(DECODE_MS_PER_TOKEN * random.uniform(0.9, 1.15) / 1000)
            itl.observe(time.perf_counter() - step_t0)
        decode_ms = (time.perf_counter() - decode_t0) * 1000

        # ---- STAGE 4: DETOKENIZATION -------------------------------------
        # Token IDs -> text, CPU-side. When streaming this runs per token and
        # must buffer partial UTF-8 sequences.
        await asyncio.sleep(completion_tokens * DETOKENIZE_MS_PER_TOK / 1000)

        # ---- STAGE 5: TEARDOWN -------------------------------------------
        # Free the KV blocks and release the slot, letting the scheduler admit
        # the next queued request.
        _kv_used -= prompt_tokens + completion_tokens
        _state["kv_cache_usage_percent"] = round(min(1.0, _kv_used / KV_CACHE_TOKENS) * 100, 2)
        _state["requests_running"] -= 1

    total_ms = (time.perf_counter() - t0) * 1000
    ttft.observe(ttft_ms / 1000)
    tpot.observe((total_ms - ttft_ms) / completion_tokens / 1000)
    e2e.observe(total_ms / 1000)
    queue_time.observe(queue_ms / 1000)
    prefill_time.observe(prefill_ms / 1000)
    decode_time.observe(decode_ms / 1000)
    inference_time.observe((prefill_ms + decode_ms) / 1000)
    _state["prompt_tokens_total"] += prompt_tokens
    _state["generation_tokens_total"] += completion_tokens
 
    return {
        "content": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "timing_ms": {
            "queue": round(queue_ms, 2),
            "ttft": round(ttft_ms, 2),
            "total": round(total_ms, 2),
            "per_output_token": round((total_ms - ttft_ms) / completion_tokens, 2),
        },
    }
 
 
# ---------------------------------------------------------------------------
# 6. SERVICE
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM Inference Simulator")
 
 
class GenerateRequest(BaseModel):
    prompt_id: int
 
 
@app.get("/metrics")
async def metrics():
    return {
        "requests_running": _state["requests_running"],
        "requests_waiting": _state["requests_waiting"],
        "kv_cache_usage_percent": _state["kv_cache_usage_percent"],
        "prompt_tokens_total": _state["prompt_tokens_total"],
        "generation_tokens_total": _state["generation_tokens_total"],
        "latency_seconds": {
            "time_to_first_token": ttft.summary(),
            "time_per_output_token": tpot.summary(),
            "inter_token_latency": itl.summary(),
            "e2e_request_latency": e2e.summary(),
            "request_queue_time": queue_time.summary(),
            "request_prefill_time": prefill_time.summary(),
            "request_decode_time": decode_time.summary(),
            "request_inference_time": inference_time.summary(),
        },
    }
 
 
@app.post("/generate")
async def generate(req: GenerateRequest):
    if not 0 <= req.prompt_id < len(PROMPTS):
        raise HTTPException(404, f"prompt_id must be in [0, {len(PROMPTS) - 1}]")
    result = await simulate_inference(PROMPTS[req.prompt_id], req.prompt_id)
    return {"prompt_id": req.prompt_id, "model": "sim-8b-instruct", **result}
 
 
@app.get("/health")
async def health():
    return {"status": "ok", "records": len(PROMPTS), "vocab": len(WORD_BANK)}
