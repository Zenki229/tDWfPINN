# Empirical complexity verification for generalized GJ Type-I/Type-II tDWfPINN

This package performs **computational** verification of the storage and training-backward FLOPs savings between generalized GJ Type-I and generalized GJ Type-II fractional residuals.

The code constructs a real bias-free `torch.nn.Module`, builds the generalized Gauss--Jacobi fractional residual graph, and compares:

- `GJ-I`: shifted quadrature nodes use time-derivative graphs, i.e. `partial_t u_theta`;
- `GJ-II`: shifted quadrature nodes use ordinary value graphs, i.e. `u_theta`.

All derivative computations use the PDE implementation in:

```text
tdw_verify/generalized_gj.py
```

There is no `repo` backend, no `generalized` backend switch, and no external PDE adapter.  All `N`, `M`, `d`, `L`, and `H` sweeps use the same generalized GJ implementation.

Default settings:

```text
alpha   = 1.5
dtype   = torch.float64
network = bias-free tanh MLP
```

The bias-free network is used to match the paper's dense-layer MAC definition:

```text
A_mac = H d + (L - 1) H^2 + H.
```

Only dense matrix multiplications and activation-related operations are counted in the theoretical FLOPs formulas; bias additions and bias-gradient reductions are not included.

---

## 1. File structure

```text
tdwfpinn_empirical_complexity_v6/
├── README.md
├── requirements.txt
├── scripts/
│   ├── test_generalized_gj.py
│   ├── run_test_gpu.sh
│   ├── run_storage_gpu.sh
│   ├── run_flops_gpu.sh
│   ├── run_all_gpu.sh
│   ├── run_storage_sweep.py
│   ├── run_flops_sweep.py
│   ├── plot_storage_orders.py
│   └── plot_flops_orders.py
└── tdw_verify/
    ├── config.py
    ├── generalized_gj.py
    ├── build_loss.py
    ├── measure.py
    ├── models.py
    ├── plotting.py
    ├── points.py
    └── sweep.py
```

The shell scripts are intentionally separated:

```bash
bash scripts/run_test_gpu.sh
bash scripts/run_storage_gpu.sh
bash scripts/run_flops_gpu.sh
```

A convenience all-in-one script is also provided:

```bash
bash scripts/run_all_gpu.sh
```

---

## 2. Installation

```bash
cd tdwfpinn_empirical_complexity_v6
python -m pip install -r requirements.txt
```

A CUDA GPU is required for storage measurement.  FLOPs profiling can run on CPU or CUDA, but the intended setting is CUDA because the storage experiment is GPU-specific.

---

## 3. Timestamped output folders

All shell entry points create timestamped output directories automatically.

For example:

```text
outputs/test_20260525_153012/
outputs/storage_20260525_153020/
outputs/flops_20260525_153045/
```

The all-in-one script creates one timestamped root folder:

```text
outputs/run_20260525_153000/test/
outputs/run_20260525_153000/storage/
outputs/run_20260525_153000/flops/
```

You may also pass an explicit output directory as the first argument:

```bash
bash scripts/run_storage_gpu.sh outputs/my_storage_run
bash scripts/run_flops_gpu.sh outputs/my_flops_run
```

---

## 4. Warm-up policy

The storage and FLOPs sweep scripts run an **unrecorded warm-up** before writing the CSV.

This is necessary because the first CUDA/autograd measurement often includes one-time initialization costs, such as:

- CUDA context creation;
- CUDA caching allocator initialization;
- cuBLAS/cuDNN lazy initialization;
- PyTorch autograd kernel initialization;
- first-time kernel compilation or dispatch overhead;
- initial memory pool expansion.

These allocations are not part of the mathematical GJ-I/GJ-II graph-storage difference.  If they are recorded, the first `GJ-I` or first `GJ-II` row can be abnormally large, especially when `base_allocated_bytes = 0`.  That first value contaminates log-log slope fitting and can make the first sweep point look much larger than the rest.

Therefore, by default:

```text
1. the code runs one unrecorded GJ-I/GJ-II warm-up pair;
2. clears gradients and CUDA peak counters;
3. starts the recorded sweep from the stable CUDA allocator state.
```

The default shell scripts use warm-up automatically:

```bash
bash scripts/run_storage_gpu.sh
bash scripts/run_flops_gpu.sh
```

At the Python level, this corresponds to:

```bash
python scripts/run_storage_sweep.py --warmup ...
python scripts/run_flops_sweep.py --warmup ...
```

For already collected old CSV files without warm-up, the storage plotting script supports:

```bash
--drop-first-pair
```

This removes the first recorded `GJ-I`/`GJ-II` pair before computing savings and slopes:

```bash
python scripts/plot_storage_orders.py \
  --csv old_storage_sweep.csv \
  --metric graph_peak_delta_bytes \
  --drop-first-pair \
  --out-dir outputs/storage_plots_drop_first
```

The option removes the first complete pair, not just a single row, so the Type-I minus Type-II saving remains well-defined.

---

## 5. Storage metrics

The storage sweep records several CUDA memory metrics.  The two most important ones are:

```text
graph_allocated_bytes
```

and

```text
graph_peak_delta_bytes
```

They are defined as:

```python
graph_allocated_bytes = memory_allocated_after_loss_build - memory_allocated_before_loss_build

graph_peak_delta_bytes = max_memory_allocated_during_loss_build - memory_allocated_before_loss_build
```

The plotted storage saving is:

```text
storage_saving = measured_storage(GJ-I) - measured_storage(GJ-II)
```

### Why the default metric is `graph_peak_delta_bytes`

The default storage plotting metric is:

```text
graph_peak_delta_bytes
```

This metric measures the **peak CUDA allocation during generalized GJ loss construction**.  It is the recommended storage metric for the paper figures because it reflects the actual peak GPU footprint that can cause out-of-memory errors.

This choice matters especially for the `d` sweep.  The predicted storage saving contains a dimension-dependent term:

```text
Delta S = O(N M d + N M L H).
```

In the implementation, the `d`-dependent tensors in `GJ-I` can appear as transient allocations during the construction of shifted time-derivative graphs.  Examples include:

```text
shifted points        ~ shape (N M, d)
full input gradients  ~ shape (N M, d)
```

Even though the final residual only needs the time component `partial_t u_theta`, autograd may temporarily allocate the full input-gradient tensor with all `d` coordinates.  Some of these tensors are released or reused before the final loss object remains alive.

Therefore:

```text
graph_allocated_bytes
```

only measures the live memory after the loss graph has been built, so it can miss or hide the transient `d`-dependent footprint.

By contrast:

```text
graph_peak_delta_bytes
```

captures the maximum allocation during graph construction, so it can reveal the `O(N M d)` contribution more clearly.

In short:

```text
graph_allocated_bytes   = retained live graph memory after loss construction
graph_peak_delta_bytes  = peak graph-construction GPU footprint
```

Use `graph_peak_delta_bytes` for the main storage scaling figures.

You can still plot retained graph memory explicitly:

```bash
python scripts/plot_storage_orders.py \
  --csv outputs/storage_<timestamp>/storage_sweep.csv \
  --metric graph_allocated_bytes \
  --out-dir outputs/storage_<timestamp>/storage_plots_allocated
```

---

## 6. Separate test script

Run a small smoke test:

```bash
bash scripts/run_test_gpu.sh
```

This writes:

```text
outputs/test_<timestamp>/test_generalized_gj.csv
```

The test checks that generalized `GJ-I` and `GJ-II` losses can be built and differentiated with the bias-free MLP.

---

## 7. Separate storage script

Run the full storage sweep:

```bash
bash scripts/run_storage_gpu.sh
```

This writes:

```text
outputs/storage_<timestamp>/storage_sweep.csv
outputs/storage_<timestamp>/storage_plots/storage_saving_vs_N.png
outputs/storage_<timestamp>/storage_plots/storage_saving_vs_M.png
outputs/storage_<timestamp>/storage_plots/storage_saving_vs_d.png
outputs/storage_<timestamp>/storage_plots/storage_saving_vs_L.png
outputs/storage_<timestamp>/storage_plots/storage_saving_vs_H.png
outputs/storage_<timestamp>/storage_plots/storage_slope_summary.csv
```

By default, the plotting metric is:

```text
graph_peak_delta_bytes
```

The output plots show the measured saving:

```text
Delta S = S_GJ-I - S_GJ-II
```

for each sweep variable.

---

## 8. Separate FLOPs script

Run the full FLOPs sweep:

```bash
bash scripts/run_flops_gpu.sh
```

This writes:

```text
outputs/flops_<timestamp>/flops_sweep.csv
outputs/flops_<timestamp>/flops_plots/flops_saving_vs_N.png
outputs/flops_<timestamp>/flops_plots/flops_saving_vs_M.png
outputs/flops_<timestamp>/flops_plots/flops_saving_vs_d.png
outputs/flops_<timestamp>/flops_plots/flops_saving_vs_L.png
outputs/flops_<timestamp>/flops_plots/flops_saving_vs_H.png
outputs/flops_<timestamp>/flops_plots/flops_slope_summary.csv
```

For FLOPs, the fractional residual graph is built first, and only the training backward pass is profiled:

```python
with torch.profiler.profile(with_flops=True):
    loss.backward()
```

The plotted FLOPs saving is:

```text
flops_saving = measured_backward_flops(GJ-I) - measured_backward_flops(GJ-II)
```

PyTorch profiler FLOPs are operator-level estimates.  They usually count dense matrix multiplications reliably.  Some elementwise autograd operations may report zero FLOPs depending on the PyTorch build, so the package also records:

```text
backward_sec
```

as a secondary runtime proxy.

---

## 9. Run everything

```bash
bash scripts/run_all_gpu.sh
```

This creates:

```text
outputs/run_<timestamp>/test/
outputs/run_<timestamp>/storage/
outputs/run_<timestamp>/flops/
```

---

## 10. Python-level commands

Storage sweep:

```bash
python scripts/run_storage_sweep.py \
  --device cuda \
  --alpha 1.5 \
  --dtype float64 \
  --warmup \
  --repeats 3 \
  --out-csv outputs/storage_sweep.csv

python scripts/plot_storage_orders.py \
  --csv outputs/storage_sweep.csv \
  --metric graph_peak_delta_bytes \
  --out-dir outputs/storage_plots
```

FLOPs sweep:

```bash
python scripts/run_flops_sweep.py \
  --device cuda \
  --alpha 1.5 \
  --dtype float64 \
  --warmup \
  --repeats 1 \
  --out-csv outputs/flops_sweep.csv

python scripts/plot_flops_orders.py \
  --csv outputs/flops_sweep.csv \
  --metric backward_profiler_flops \
  --out-dir outputs/flops_plots
```

Custom storage sweeps:

```bash
python scripts/run_storage_sweep.py \
  --sweeps N,M \
  --N-values 16,32,64,128 \
  --M-values 4,8,16,32 \
  --device cuda \
  --warmup

python scripts/run_storage_sweep.py \
  --sweeps d \
  --d-values 2,4,8,16,32,64,128 \
  --device cuda \
  --warmup
```

---

## 11. Expected scaling

The measured storage saving is expected to follow:

```text
Delta S = O(N M d + N M L H).
```

With fp64 tensor data, the theoretical storage saving used in the paper is:

```text
Delta S = 8 N M (d + 2 L H) bytes.
```

The measured backward-FLOPs saving is expected to follow:

```text
Delta F = N M [4{H d + (L - 1)H^2 + H} + 5 L H].
```

Therefore:

- storage saving is approximately linear in `N`, `M`, `d`, `L`, and `H`;
- backward-FLOPs saving is linear in `N`, `M`, and `d`;
- backward-FLOPs saving can grow quadratically in `H` when the dense hidden-to-hidden term `(L - 1)H^2` dominates;
- for small `d`, the term `2 L H` can dominate `d + 2 L H`, so a small `d` sweep may look nearly flat unless `d` is large enough or `L,H` are reduced.

For the `d` sweep, `graph_peak_delta_bytes` is preferred over `graph_allocated_bytes` because it captures transient dimension-dependent allocations during graph construction.

---

## 12. Notes on interpreting the plots

Each plot uses measured data, not theoretical formula values.

The plotting scripts aggregate repeated measurements by median.  Median aggregation is used because CUDA memory and profiler measurements can contain occasional allocator or kernel-dispatch outliers.

For log-log slope fitting, the script fits:

```text
log(Delta) = a log(x) + b
```

where `a` is the fitted scaling exponent.

If a sweep variable has an additive background term, the log-log slope may be smaller than the asymptotic theoretical exponent for small values.  This is especially relevant for the `d` sweep, since:

```text
Delta S = 8 N M (d + 2 L H).
```

When `d << 2 L H`, the storage saving is dominated by `2 L H`, so the curve may look almost constant in `d`.  Increasing the `d` range or decreasing `L,H` makes the `O(d)` contribution easier to observe.
