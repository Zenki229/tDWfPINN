"""Measurement primitives for graph storage and backward FLOPs."""
from __future__ import annotations

import gc
import time
from dataclasses import asdict
from typing import Any, Literal

import torch

from .build_loss import build_gj_loss_from_components, prepare_gj_components
from .config import BenchConfig

MethodName = Literal["GJ-I", "GJ-II"]


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def measure_graph_storage(
    cfg: BenchConfig,
    method: MethodName,
    device: torch.device,
    do_backward: bool = True,
) -> dict[str, Any]:
    """Measure retained graph allocation after constructing the generalized GJ loss.

    On CUDA, the baseline is taken after allocating static objects.  Thus
    ``graph_allocated_bytes`` measures the allocation increment caused by the
    fractional residual autograd graph, not by model parameters or quadrature nodes.
    """
    _cleanup(device)
    torch.set_default_dtype(torch.float64 if cfg.dtype == "float64" else torch.float32)

    model, points, operator = prepare_gj_components(cfg, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        base_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        base_allocated = 0

    start_build = time.perf_counter()
    loss = build_gj_loss_from_components(cfg, method=method, model=model, points=points, operator=operator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    build_sec = time.perf_counter() - start_build

    if device.type == "cuda":
        after_build = torch.cuda.memory_allocated(device)
        graph_allocated = max(0, after_build - base_allocated)
        graph_peak_delta = max(0, torch.cuda.max_memory_allocated(device) - base_allocated)
    else:
        graph_allocated = float("nan")
        graph_peak_delta = float("nan")

    backward_sec = float("nan")
    backward_peak_delta = float("nan")
    if do_backward:
        start_backward = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_sec = time.perf_counter() - start_backward
        if device.type == "cuda":
            backward_peak_delta = max(0, torch.cuda.max_memory_allocated(device) - base_allocated)

    result = {
        **asdict(cfg),
        "method": method,
        "device": str(device),
        "base_allocated_bytes": base_allocated,
        "graph_allocated_bytes": graph_allocated,
        "graph_peak_delta_bytes": graph_peak_delta,
        "backward_peak_delta_bytes": backward_peak_delta,
        "build_sec": build_sec,
        "backward_sec": backward_sec,
    }

    del loss, model, points, operator
    _cleanup(device)
    return result


def measure_backward_flops(cfg: BenchConfig, method: MethodName, device: torch.device) -> dict[str, Any]:
    """Measure PyTorch profiler-reported FLOPs for ``loss.backward()``."""
    _cleanup(device)
    torch.set_default_dtype(torch.float64 if cfg.dtype == "float64" else torch.float32)

    model, points, operator = prepare_gj_components(cfg, device=device)
    loss = build_gj_loss_from_components(cfg, method=method, model=model, points=points, operator=operator)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]

    start = time.perf_counter()
    with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False, profile_memory=False) as prof:
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed_sec = time.perf_counter() - start

    flops = 0
    for event in prof.key_averages():
        if event.flops is not None:
            flops += int(event.flops)

    result = {
        **asdict(cfg),
        "method": method,
        "device": str(device),
        "backward_profiler_flops": flops,
        "backward_sec": elapsed_sec,
    }

    del loss, model, points, operator, prof
    _cleanup(device)
    return result
