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
 
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Gauge, Histogram,
                               Counter as PromCounter, generate_latest)
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
# 4. METRICS -- same names vLLM exposes, so stock dashboards work unchanged.
# ---------------------------------------------------------------------------
running = Gauge("vllm:num_requests_running", "Requests running on GPU")
waiting = Gauge("vllm:num_requests_waiting", "Requests queued")
kv_usage = Gauge("vllm:kv_cache_usage_perc", "Fraction of KV blocks in use")
ttft = Histogram("vllm:time_to_first_token_seconds", "TTFT",
                 buckets=[.01, .05, .1, .25, .5, 1, 2.5, 5, 10])
e2e = Histogram("vllm:e2e_request_latency_seconds", "End-to-end latency")
ptoks = PromCounter("vllm:prompt_tokens_total", "Prefill tokens processed")
gtoks = PromCounter("vllm:generation_tokens_total", "Generation tokens produced")
 
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
    waiting.inc()
    async with _slots:
        waiting.dec()
        running.inc()
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
        kv_usage.set(min(1.0, _kv_used / KV_CACHE_TOKENS))
 
        # ---- STAGE 2: PREFILL --------------------------------------------
        # One parallel forward pass over the whole prompt. Compute-bound, so
        # cost scales LINEARLY WITH PROMPT LENGTH. Fills the KV cache and
        # emits the first token -- this is what determines TTFT.
        await asyncio.sleep(prompt_tokens * PREFILL_MS_PER_TOKEN / 1000)
        ttft_ms = (time.perf_counter() - t0) * 1000
 
        # ---- STAGE 3: DECODE ---------------------------------------------
        # Autoregressive loop, one token per iteration. Memory-bandwidth-bound:
        # every step re-reads all model weights, so per-token cost is roughly
        # CONSTANT and INDEPENDENT OF PROMPT LENGTH -- that is what the KV
        # cache buys us. Jitter models sampling and scheduler variance.
        for _ in range(completion_tokens):
            await asyncio.sleep(DECODE_MS_PER_TOKEN * random.uniform(0.9, 1.15) / 1000)
 
        # ---- STAGE 4: DETOKENIZATION -------------------------------------
        # Token IDs -> text, CPU-side. When streaming this runs per token and
        # must buffer partial UTF-8 sequences.
        await asyncio.sleep(completion_tokens * DETOKENIZE_MS_PER_TOK / 1000)
 
        # ---- STAGE 5: TEARDOWN -------------------------------------------
        # Free the KV blocks and release the slot, letting the scheduler admit
        # the next queued request.
        _kv_used -= prompt_tokens + completion_tokens
        kv_usage.set(min(1.0, _kv_used / KV_CACHE_TOKENS))
        running.dec()
 
    total_ms = (time.perf_counter() - t0) * 1000
    ttft.observe(ttft_ms / 1000)
    e2e.observe(total_ms / 1000)
    ptoks.inc(prompt_tokens)
    gtoks.inc(completion_tokens)
 
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
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
 
 
@app.post("/generate")
async def generate(req: GenerateRequest):
    if not 0 <= req.prompt_id < len(PROMPTS):
        raise HTTPException(404, f"prompt_id must be in [0, {len(PROMPTS) - 1}]")
    result = await simulate_inference(PROMPTS[req.prompt_id], req.prompt_id)
    return {"prompt_id": req.prompt_id, "model": "sim-8b-instruct", **result}
 
 
@app.get("/health")
async def health():
    return {"status": "ok", "records": len(PROMPTS), "vocab": len(WORD_BANK)}
