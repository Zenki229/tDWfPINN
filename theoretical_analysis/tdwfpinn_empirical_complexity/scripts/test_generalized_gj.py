#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv

import torch

from tdw_verify.config import BenchConfig
from tdw_verify.measure import measure_backward_flops, measure_graph_storage


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test generalized_gj.py GJ-I/GJ-II autograd graphs.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--L", type=int, default=1)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = BenchConfig(N=args.N, M=args.M, d=args.d, L=args.L, H=args.H, alpha=args.alpha, dtype=args.dtype)
    rows = []
    for method in ["GJ-I", "GJ-II"]:
        storage = measure_graph_storage(cfg, method=method, device=device, do_backward=True)
        flops = measure_backward_flops(cfg, method=method, device=device)
        row = {**storage, "backward_profiler_flops": flops["backward_profiler_flops"], "profiler_backward_sec": flops["backward_sec"]}
        rows.append(row)
        print(
            f"{method}: graph_allocated_bytes={row['graph_allocated_bytes']} "
            f"backward_peak_delta_bytes={row['backward_peak_delta_bytes']} "
            f"backward_profiler_flops={row['backward_profiler_flops']}"
        )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
