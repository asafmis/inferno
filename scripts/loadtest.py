"""Send POST /generate requests against a running simulator instance.

Prompts are sent in the order they appear in the dataset (original_id column
order), using each row's index as prompt_id -- the only value the /generate
endpoint accepts (see test.py's `0 <= prompt_id < len(PROMPTS)` check).

Requires the server to already be running:
    uvicorn test:app --reload

Usage:
    python scripts/loadtest.py [--count N] [--host URL]
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sharegpt-prompts-1k.csv"
DEFAULT_HOST = "http://localhost:8000"
DEFAULT_COUNT = 5


def load_original_ids(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [row["original_id"] for row in csv.DictReader(f)]


def send_requests(count: int = DEFAULT_COUNT, host: str = DEFAULT_HOST) -> None:
    original_ids = load_original_ids(DATA_PATH)
    for prompt_id, original_id in enumerate(original_ids[:count]):
        payload = {"prompt_id": prompt_id}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        print(
            f"curl -X POST {host.replace('http://', '')}/generate "
            f'-H "content-type: application/json" -d \'{json.dumps(payload)}\''
        )
        try:
            with urllib.request.urlopen(req) as resp:
                print(json.dumps(json.loads(resp.read()), indent=2))
        except urllib.error.HTTPError as e:
            print(json.dumps(json.loads(e.read()), indent=2))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="number of requests to send")
    parser.add_argument("--host", default=DEFAULT_HOST, help="base URL of the running server")
    args = parser.parse_args()
    send_requests(args.count, args.host)


if __name__ == "__main__":
    main()
