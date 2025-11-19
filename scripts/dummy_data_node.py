#!/usr/bin/env python3
"""
Process raw `.txt` files into an LLMSR-ready benchmark folder.

The script acts as a "dummy data node": it ingests every text file under
`raw_data/`, cleans and concatenates them, optionally performs min-max
normalization, and finally writes `train.csv`, `test_id.csv`, and
`test_ood.csv` under `data/<problem_name>/`.

Example:
    python scripts/dummy_data_node.py \
        --input-dir raw_data \
        --output-root data \
        --problem-name dummy_benchmark \
        --train-frac 0.7 \
        --test-id-frac 0.15 \
        --normalize
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

TARGET_CANDIDATES = ("target", "label", "y", "output")


def parse_txt(path: Path) -> pd.DataFrame:
    """Parse a text file into a numeric DataFrame."""
    attempts = [
        {"sep": None, "engine": "python", "header": 0},
        {"delim_whitespace": True, "header": 0},
        {"sep": None, "engine": "python", "header": None},
        {"delim_whitespace": True, "header": None},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, comment="#", **kwargs)
        except Exception as exc:  # pragma: no cover - pandas errors vary
            last_error = exc
            continue
        if df.empty:
            continue
        return df
    raise ValueError(f"Unable to parse {path}. Last error: {last_error}")


def sanitize_numeric(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Ensure dataframe is numeric and has at least 2 columns."""
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")
    if df.empty:
        raise ValueError(f"{source} produced no usable rows after dropping NaNs.")

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=0, how="any")
    if numeric_df.shape[1] < 2:
        raise ValueError(f"{source} needs at least two numeric columns.")
    return numeric_df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Move the target column to the end and rename columns to LLMSR schema."""
    cols = list(df.columns)
    target_col = None
    for candidate in TARGET_CANDIDATES:
        for col in cols:
            if str(col).lower() == candidate:
                target_col = col
                break
        if target_col:
            break
    if target_col is None:
        target_col = cols[-1]

    feature_cols = [c for c in cols if c != target_col]
    ordered = df[feature_cols + [target_col]].copy()

    rename_map = {col: f"feature_{idx}" for idx, col in enumerate(feature_cols)}
    rename_map[target_col] = "target"
    ordered = ordered.rename(columns=rename_map)
    return ordered


def load_raw_tables(input_dir: Path) -> pd.DataFrame:
    """Concatenate every .txt table inside `input_dir`."""
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found under {input_dir}.")

    tables: List[pd.DataFrame] = []
    for file in files:
        raw = parse_txt(file)
        numeric = sanitize_numeric(raw, file)
        ordered = reorder_columns(numeric)
        tables.append(ordered)
        print(f"Loaded {file} -> {ordered.shape[0]} rows, {ordered.shape[1]} columns")

    combined = pd.concat(tables, ignore_index=True)
    print(f"Combined dataset shape: {combined.shape}")
    return combined


def normalize_min_max(df: pd.DataFrame) -> pd.DataFrame:
    """Scale every column to [0, 1]. Constant columns become 0."""
    scaled = df.copy()
    for col in scaled.columns:
        col_min = scaled[col].min()
        col_max = scaled[col].max()
        if col_max > col_min:
            scaled[col] = (scaled[col] - col_min) / (col_max - col_min)
        else:
            scaled[col] = 0.0
    return scaled


def split_dataset(
    df: pd.DataFrame, train_frac: float, test_id_frac: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split the dataset into train/test_id/test_ood."""
    if not 0 < train_frac < 1:
        raise ValueError("`train_frac` must be between 0 and 1.")
    if not 0 < test_id_frac < 1:
        raise ValueError("`test_id_frac` must be between 0 and 1.")
    if train_frac + test_id_frac >= 1:
        raise ValueError("`train_frac + test_id_frac` must be less than 1.")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(round(n * train_frac))
    test_id_end = train_end + int(round(n * test_id_frac))

    train = shuffled.iloc[:train_end]
    test_id = shuffled.iloc[train_end:test_id_end]
    test_ood = shuffled.iloc[test_id_end:]

    # Ensure none of the splits are empty; fallback by borrowing rows if needed.
    if train.empty or test_id.empty or test_ood.empty:
        raise ValueError(
            "One of the splits is empty. Please adjust the fractions or "
            "provide more data."
        )

    return train, test_id, test_ood


def save_splits(train: pd.DataFrame, test_id: pd.DataFrame, test_ood: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test_id.to_csv(out_dir / "test_id.csv", index=False)
    test_ood.to_csv(out_dir / "test_ood.csv", index=False)
    print(f"Saved splits to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Dummy data node for LLMSR benchmarks.")
    parser.add_argument("--input-dir", type=Path, default=Path("raw_data"), help="Directory containing .txt files.")
    parser.add_argument("--output-root", type=Path, default=Path("data"), help="Base directory for processed benchmarks.")
    parser.add_argument("--problem-name", type=str, default="dummy_node", help="Name of the benchmark folder to create.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Fraction of rows assigned to train.csv.")
    parser.add_argument("--test-id-frac", type=float, default=0.15, help="Fraction of rows assigned to test_id.csv.")
    parser.add_argument("--normalize", action="store_true", help="Enable min-max normalization for all columns.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling before splitting.")

    args = parser.parse_args()

    df = load_raw_tables(args.input_dir)
    if args.normalize:
        df = normalize_min_max(df)
        print("Applied min-max normalization to all columns.")

    train, test_id, test_ood = split_dataset(df, args.train_frac, args.test_id_frac, args.seed)
    output_dir = args.output_root / args.problem_name
    save_splits(train, test_id, test_ood, output_dir)


if __name__ == "__main__":
    main()

