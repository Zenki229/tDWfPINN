"""Sweep helpers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import torch

from .config import BenchConfig, DEFAULT_SWEEP_VALUES


def _warmup_once(measure_fn: Callable, baseline: BenchConfig, device: torch.device) -> None:
    """Run unrecorded warm-up measurements before the real sweep.

    The first CUDA autograd measurement can include one-time CUDA context,
    allocator, cuBLAS, and autograd-kernel initialization costs.  Those costs
    are unrelated to the GJ-I/GJ-II scaling law, so this pass deliberately
    triggers them without writing the result to the CSV.
    """
    warm_cfg = baseline.with_update(
        N=min(max(baseline.N, 1), 8),
        M=min(max(baseline.M, 1), 4),
        d=max(baseline.d, 2),
        L=1,
        H=min(max(baseline.H, 2), 8),
        seed=baseline.seed - 1,
    )
    print(
        "Running unrecorded CUDA/autograd warm-up "
        f"with N={warm_cfg.N}, M={warm_cfg.M}, d={warm_cfg.d}, L={warm_cfg.L}, H={warm_cfg.H}",
        flush=True,
    )
    for method in ("GJ-I", "GJ-II"):
        _ = measure_fn(warm_cfg, method=method, device=device)
    print("Warm-up finished; starting recorded sweep.", flush=True)


def run_pair_sweep(
    measure_fn: Callable,
    baseline: BenchConfig,
    sweep_vars: list[str],
    sweep_values: dict[str, list[int]],
    device: torch.device,
    repeats: int,
    out_csv: str | Path,
    warmup: bool = True,
) -> None:
    if warmup:
        _warmup_once(measure_fn, baseline, device)

    rows: list[dict] = []
    for sweep_var in sweep_vars:
        values = sweep_values.get(sweep_var, DEFAULT_SWEEP_VALUES[sweep_var])
        for value in values:
            cfg = baseline.with_update(**{sweep_var: int(value)})
            for repeat in range(repeats):
                cfg_rep = cfg.with_update(seed=baseline.seed + repeat * 1009)
                for method in ("GJ-I", "GJ-II"):
                    row = measure_fn(cfg_rep, method=method, device=device)
                    row["sweep_var"] = sweep_var
                    row["sweep_value"] = int(value)
                    row["repeat"] = repeat
                    rows.append(row)
                    print(f"{sweep_var}={value} repeat={repeat} method={method} done", flush=True)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows were generated")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
