from pathlib import Path

import pandas as pd


def iter_csv_runs(folder: Path, prefix: str, required_columns: set[str]):
    for path in sorted(folder.glob(f"{prefix}*.csv")):
        df = pd.read_csv(path)
        missing = required_columns - set(df.columns)
        if missing:
            print(f"[WARN] Ignored {path}: missing {sorted(missing)}")
            continue

        if "repetition_id" not in df.columns:
            df["repetition_id"] = 0

        for repetition_id, run in df.groupby("repetition_id", sort=True):
            yield path, repetition_id, run.sort_values("time").copy()
