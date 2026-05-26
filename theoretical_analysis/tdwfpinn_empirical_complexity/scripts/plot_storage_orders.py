#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from tdw_verify.plotting import plot_order_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Plot storage saving order from CSV.")
    p.add_argument("--csv", type=str, default="outputs/storage_sweep.csv")
    p.add_argument("--out-dir", type=str, default="outputs/storage_plots")
    p.add_argument("--drop-first-pair", action="store_true", help="Drop the first recorded sweep/repeat pair before plotting an old CSV collected without warm-up.")
    p.add_argument(
        "--metric",
        choices=["graph_allocated_bytes", "graph_peak_delta_bytes", "backward_peak_delta_bytes"],
        default="graph_allocated_bytes",
    )
    args = p.parse_args()
    plot_order_csv(args.csv, metric=args.metric, kind="storage", out_dir=args.out_dir, drop_first_pair=args.drop_first_pair)


if __name__ == "__main__":
    main()
