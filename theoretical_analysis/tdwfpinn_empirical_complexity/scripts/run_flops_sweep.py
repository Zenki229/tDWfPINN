#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import torch

from tdw_verify.config import BenchConfig, DEFAULT_SWEEP_VALUES, parse_int_list
from tdw_verify.measure import measure_backward_flops
from tdw_verify.sweep import run_pair_sweep


def main() -> None:
    p = argparse.ArgumentParser(description="Measure profiler-reported backward FLOPs for generalized GJ-I vs GJ-II.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-csv", type=str, default="outputs/flops_sweep.csv")
    p.add_argument("--sweeps", type=str, default="N,M,d,L,H")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--N-values", type=str, default=None)
    p.add_argument("--M-values", type=str, default=None)
    p.add_argument("--d-values", type=str, default=None)
    p.add_argument("--L-values", type=str, default=None)
    p.add_argument("--H-values", type=str, default=None)
    warmup_group = p.add_mutually_exclusive_group()
    warmup_group.add_argument("--warmup", dest="warmup", action="store_true", help="Run an unrecorded CUDA/autograd warm-up before measurements. This is the default.")
    warmup_group.add_argument("--no-warmup", dest="warmup", action="store_false", help="Disable the unrecorded warm-up pass.")
    p.set_defaults(warmup=True)
    args = p.parse_args()

    baseline = BenchConfig(N=args.N, M=args.M, d=args.d, L=args.L, H=args.H, alpha=args.alpha, dtype=args.dtype)
    sweep_values = dict(DEFAULT_SWEEP_VALUES)
    for name in ["N", "M", "d", "L", "H"]:
        value_text = getattr(args, f"{name}_values")
        if value_text:
            sweep_values[name] = parse_int_list(value_text)
    sweep_vars = [x.strip() for x in args.sweeps.split(",") if x.strip()]
    device = torch.device(args.device)
    run_pair_sweep(
        measure_fn=measure_backward_flops,
        baseline=baseline,
        sweep_vars=sweep_vars,
        sweep_values=sweep_values,
        device=device,
        repeats=args.repeats,
        out_csv=args.out_csv,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
