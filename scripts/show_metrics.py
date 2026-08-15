"""Fetch GET /metrics from a running simulator instance and pretty-print it.

Requires the server to already be running:
    uvicorn test:app --reload

Usage:
    python scripts/show_metrics.py [--host URL]
"""

from __future__ import annotations

import argparse
import json
import urllib.request

DEFAULT_HOST = "http://localhost:8000"


def show_metrics(host: str = DEFAULT_HOST) -> None:
    with urllib.request.urlopen(f"{host}/metrics") as resp:
        print(json.dumps(json.loads(resp.read()), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="base URL of the running server")
    args = parser.parse_args()
    show_metrics(args.host)


if __name__ == "__main__":
    main()
