"""Compare cheap token-count approximations against a real BPE tokenizer.

This script measures how far six such cheap approximations drift from ground
truth (tiktoken's cl100k_base, the BPE vocab used by GPT-3.5/GPT-4) on a
real prompt distribution.

Usage:
    python scripts/estimate_tokens.py [--sample N]
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path

import pandas as pd
import tiktoken

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sharegpt-prompts-1k.csv"
TEXT_COLUMN = "original_prompt"


# --- approximation formulas under test -------------------------------------
#
# Each takes raw text and returns an estimated token count. They trade
# accuracy for how much of the text they have to look at: #1 only counts
# characters, #4 is the only one that inspects each word's length.


def chars_over_4(text: str) -> int:
    """#1 Character ratio: OpenAI's ~4 chars/token rule of thumb."""
    return max(1, math.ceil(len(text) / 4))


def words_times_133(text: str) -> int:
    """#2 Word ratio: from the companion rule, 100 tokens ~= 75 words."""
    return max(1, math.ceil(len(text.split()) * 1.33))

# atom - a unit of text, such as a single word, number, or punctuation mark
_ATOM_RE = re.compile(r"\w+|[^\w\s]")


def atoms_times_11(text: str) -> int:
    """#3 Word-and-punctuation atoms with a flat correction factor."""
    atoms = _ATOM_RE.findall(text)
    return max(1, math.ceil(len(atoms) * 1.1))


def subword_aware(text: str) -> int:
    """#4 Like #3, but charges long alphanumeric atoms proportionally more."""
    total = 0
    for atom in _ATOM_RE.findall(text):
        total += max(1, math.ceil(len(atom) / 5)) if atom.isalnum() else 1
    return max(1, total)


def hybrid_floor(text: str) -> int:
    """#5 chars/4, floored at one token per whitespace-separated word."""
    return max(1, math.ceil(len(text) / 4), len(text.split()))


def char_class_weighted(text: str) -> int:
    """#6 Weights ASCII and non-ASCII characters differently for multilingual text."""
    ascii_chars = sum(c.isascii() for c in text)
    other = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 4 + other / 1.5))


FORMULAS = {
    "1 chars/4": chars_over_4,
    "2 words*1.33": words_times_133,
    "3 atoms*1.1": atoms_times_11,
    "4 subword-aware": subword_aware,
    "5 hybrid floor": hybrid_floor,
    "6 char-class weighted": char_class_weighted,
}


def load_texts(sample: int | None) -> list[str]:
    df = pd.read_csv(DATA_PATH)
    texts = df[TEXT_COLUMN].dropna().astype(str).tolist()
    if sample:
        texts = texts[:sample]
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None, help="only use the first N prompts")
    args = parser.parse_args()

    texts = load_texts(args.sample)
    enc = tiktoken.get_encoding("cl100k_base")

    truths = [(t, len(enc.encode(t))) for t in texts]
    truths = [(t, n) for t, n in truths if n]  # drop empties once

    print(f"dataset: {DATA_PATH.name}  prompts: {len(truths)}\n")
    print(f"{'formula':22s} {'MAPE':>8s} {'median':>8s} {'max':>8s}")
    print("-" * 50)

    for name, fn in FORMULAS.items():
        errs = [abs(fn(t) - n) / n for t, n in truths]
        # MAPE - Mean Absolute Percentage Error
        mape = statistics.mean(errs)
        median = statistics.median(errs)
        worst = max(errs)
        print(f"{name:22s} {mape:8.1%} {median:8.1%} {worst:8.1%}")


if __name__ == "__main__":
    main()
