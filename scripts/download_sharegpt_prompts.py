"""Download the sharegpt-prompts-1k Kaggle dataset, clean it, and save it locally.

Requires Kaggle API credentials (kaggle.json in ~/.kaggle/, or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables).
"""

from pathlib import Path

import kagglehub
import pandas as pd

DATASET_HANDLE = "austinfairbanks/sharegpt-prompts-1k/versions/3"
DROP_COLUMNS = ["context", "prompt"]
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sharegpt-prompts-1k.csv"


def main() -> None:
    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    df = pd.read_csv(csv_files[0])

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.dropna()

    if "original_id" in df.columns:
        df = df.drop_duplicates(subset="original_id")
        assert df["original_id"].is_unique, "original_id still has duplicates"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
