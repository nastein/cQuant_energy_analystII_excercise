#!/usr/bin/env python3
"""
main.py — cQuant programming exercise 

Run:
  python main.py

Required packages are in requirements.txt
They can be installed using pip install -r requirements.txt

How to use:
- Put the downloaded data file(s) in ./data
- Fill in the TASK sections once you see the PDF
- All outputs go to ./output (CSVs + PNGs)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========
# CONFIG
# =========
RAW_DIR = Path("data/raw")
OUT_DIR = Path("output")
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"


def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data(filename):
    print('Reading data from f"{filename}" and transforming to dataframe:')
    return pd.read_csv(filename)


def save_table(df: pd.DataFrame, name: str):
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def save_fig(filename):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{filename}.png", dpi=175, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()
    #df = load_data()

    # =========
    # QUICK AUDIT (optional but helpful)
    # =========
    #audit = pd.DataFrame({
    #    "column": df.columns,
    #    "dtype": [str(df[c].dtype) for c in df.columns],
    #    "missing": [int(df[c].isna().sum()) for c in df.columns],
    #    "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    #})
    #save_table(audit, "data_audit")

    # =========
    # TASK 1 (fill in)
    # =========
    # Example pattern:
    # result = df.groupby(...).agg(...).reset_index()
    # save_table(result, "task01_some_table")
    #
    # plt.figure()
    # plt.plot(...)
    # plt.title("Task 1 ...")
    # plt.xlabel("...")
    # plt.ylabel("...")
    # save_fig("task01_some_plot")

    # =========
    # TASK 2 (fill in)
    # =========

    # =========
    # TASK 3 (fill in)
    # =========

    # =========
    # BONUS (optional)
    # =========

    print("Done. Outputs saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
