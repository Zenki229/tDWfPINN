import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from libs.jax_pde_burgers import JAXDWBurgers
from libs.jax_pde_forward import JAXDWForward
from libs.jax_pde_irregular import JAXIrregularHoleDW, JAXLShapeDW
from libs.jax_pinn import create_train_state
from libs.jax_sample import IrregularHoleSampler, LShapeSampler, TimeSpaceEasySampler
from libs.jax_utils import replicate, unreplicate


PAPER_COLORMAP = "jet"


def model_cfg(input_dim, hidden_dim=32, num_layers=3):
    return OmegaConf.create({
        "arch_name": "mlp",
        "input_dim": input_dim,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "out_dim": 1,
        "activation": "tanh",
    })


def optim_cfg():
    return OmegaConf.create({
        "learning_rate": 1e-3,
        "decay_steps": 1000,
        "decay_rate": 0.95,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "grad_accum_steps": 1,
    })


def weighting_cfg():
    return OmegaConf.create({
        "scheme": "none",
        "init_weights": {"in": 1.0, "bd": 1.0, "init": 1.0, "init_dt": 1.0},
        "momentum": 0.9,
    })


def rel_l2(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    return float(np.sqrt(np.sum((pred - true) ** 2) / (np.sum(true ** 2) + 1e-12)))


def rel_l2_or_nan(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    denom = np.sum(true ** 2)
    if denom < 1e-12:
        return np.nan
    return float(np.sqrt(np.sum((pred - true) ** 2) / denom))


def format_rel(value):
    if np.isnan(value):
        return "n/a (zero true)"
    return f"{value:.4e}"


def predict(apply_fn, params, points, chunk_size=2048):
    vals = []
    for start in range(0, len(points), chunk_size):
        chunk = jnp.asarray(points[start:start + chunk_size])
        vals.append(np.asarray(jax.device_get(apply_fn(params, chunk))).reshape(-1))
    return np.concatenate(vals, axis=0)


def train_smoke(pde, sampler, input_dim, steps, seed):
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    state = create_train_state(model_key, model_cfg(input_dim), optim_cfg(), weighting_cfg())
    state = replicate(state)
    keys = jax.random.split(key, jax.local_device_count())

    last_loss = np.nan
    for _ in range(steps):
        batch = next(sampler)
        state, loss_val, aux, keys = pde.step(state, batch, keys)
        last_loss = float(jnp.mean(loss_val))

    return unreplicate(state), last_loss


def plot_1d_case(case_name, t_grid, x_grid, true_grid, pred_grid, outdir):
    err_grid = np.abs(pred_grid - true_grid)
    rel_err = rel_l2(pred_grid, true_grid)
    vmin = float(np.nanmin([np.nanmin(true_grid), np.nanmin(pred_grid)]))
    vmax = float(np.nanmax([np.nanmax(true_grid), np.nanmax(pred_grid)]))
    err_vmax = float(np.nanmax(err_grid))

    panels = [
        ("true", "true", true_grid, vmin, vmax, PAPER_COLORMAP),
        ("sol", "sol", pred_grid, vmin, vmax, PAPER_COLORMAP),
        ("abs_error", f"abs error\nrelative error = {rel_err:.4e}",
         err_grid, 0.0, err_vmax, PAPER_COLORMAP),
    ]
    paths = []
    for suffix, title, values, lo, hi, cmap in panels:
        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        pcm = ax.pcolormesh(t_grid, x_grid, values, shading="auto", cmap=cmap,
                            vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        fig.colorbar(pcm, ax=ax, format="%.2e")
        path = outdir / f"{case_name}_{suffix}_smoke.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths, rel_err


def parse_time_slices(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def plot_2d_time_slice_files(case_name, x_grid, y_grid, true_grids, pred_grids,
                             times, outdir):
    true_stack = np.asarray(true_grids)
    pred_stack = np.asarray(pred_grids)
    valid = np.isfinite(true_stack) & np.isfinite(pred_stack)
    rel_err = rel_l2(pred_stack[valid], true_stack[valid])

    vmin = float(np.nanmin([np.nanmin(true_stack), np.nanmin(pred_stack)]))
    vmax = float(np.nanmax([np.nanmax(true_stack), np.nanmax(pred_stack)]))
    paths = []

    for time_value, true_grid, pred_grid in zip(times, true_stack, pred_stack):
        err_grid = np.abs(pred_grid - true_grid)
        row_valid = np.isfinite(true_grid) & np.isfinite(pred_grid)
        row_rel = rel_l2_or_nan(pred_grid[row_valid], true_grid[row_valid])
        err_vmax = float(np.nanmax(err_grid))
        time_token = f"t{time_value:.3f}".replace(".", "p")
        panels = [
            ("true", f"true, t={time_value:.3f}", true_grid, vmin, vmax, PAPER_COLORMAP),
            ("sol", f"sol, t={time_value:.3f}", pred_grid, vmin, vmax, PAPER_COLORMAP),
            ("abs_error",
             f"abs error, t={time_value:.3f}\nrelative error = {format_rel(row_rel)}",
             err_grid, 0.0, err_vmax, PAPER_COLORMAP),
        ]
        for suffix, title, values, lo, hi, cmap in panels:
            fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
            pcm = ax.pcolormesh(x_grid, y_grid, values, shading="auto", cmap=cmap,
                                vmin=lo, vmax=hi)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            fig.colorbar(pcm, ax=ax, format="%.2e")
            path = outdir / f"{case_name}_{time_token}_{suffix}_smoke.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)

    return paths, rel_err


def make_burgers(args):
    data = np.load(args.burgers_data)
    t_full, x_full = data["t"], data["x"]
    cfg = OmegaConf.create({
        "al": 1.5,
        "beta": 2.0,
        "xlim": [[float(x_full[0]), float(x_full[-1])]],
        "tlim": [float(t_full[0]), float(t_full[-1])],
        "method": "GJ-II",
        "GJ": {"nums": args.quad},
        "MC": {"nums": args.quad, "eps": 1e-8},
    })
    pde = JAXDWBurgers(cfg, weighting_cfg())
    sampler = TimeSpaceEasySampler(
        cfg.xlim,
        cfg.tlim,
        {"in": args.batch_in, "bd": args.batch_bd, "init": args.batch_init},
        n_devices=jax.local_device_count(),
        shard=True,
    )
    state, loss = train_smoke(pde, sampler, 2, args.steps, args.seed)

    u_full = data["u"].T
    ti = np.unique(np.linspace(0, len(t_full) - 1, args.grid_1d, dtype=int))
    xi = np.unique(np.linspace(0, len(x_full) - 1, args.grid_1d, dtype=int))
    t, x = t_full[ti], x_full[xi]
    t_grid, x_grid = np.meshgrid(t, x)
    true = u_full[np.ix_(xi, ti)]
    points = np.stack([t_grid.ravel(), x_grid.ravel()], axis=1)
    pred = predict(state.apply_fn, state.params, points).reshape(t_grid.shape)
    paths, rel_err = plot_1d_case("burgers", t_grid, x_grid, true, pred, args.outdir)
    return {
        "case": "burgers",
        "path": ";".join(str(path) for path in paths),
        "relative_error": rel_err,
        "loss": loss,
    }


def make_forward(args):
    cfg = OmegaConf.create({
        "al": 1.75,
        "k": 1,
        "lam": 1.0,
        "a": 1.0,
        "b": -0.5,
        "xlim": [[0, 1]],
        "tlim": [0, 2],
        "method": "GJ-II",
        "GJ": {"nums": args.quad},
        "MC": {"nums": args.quad, "eps": 1e-8},
    })
    pde = JAXDWForward(cfg, weighting_cfg())
    sampler = TimeSpaceEasySampler(
        cfg.xlim,
        cfg.tlim,
        {"in": args.batch_in, "bd": args.batch_bd, "init": args.batch_init},
        n_devices=jax.local_device_count(),
        shard=True,
    )
    state, loss = train_smoke(pde, sampler, 2, args.steps, args.seed)

    t = np.linspace(cfg.tlim[0], cfg.tlim[1], args.grid_1d)
    x = np.linspace(cfg.xlim[0][0], cfg.xlim[0][1], args.grid_1d)
    t_grid, x_grid = np.meshgrid(t, x)
    points = np.stack([t_grid.ravel(), x_grid.ravel()], axis=1)
    true = pde.exact(points).reshape(t_grid.shape)
    pred = predict(state.apply_fn, state.params, points).reshape(t_grid.shape)
    paths, rel_err = plot_1d_case("forward", t_grid, x_grid, true, pred, args.outdir)
    return {
        "case": "forward",
        "path": ";".join(str(path) for path in paths),
        "relative_error": rel_err,
        "loss": loss,
    }


def make_irregular_hole(args):
    cfg = OmegaConf.create({
        "al": 1.5,
        "tlim": [0, 1],
        "method": "GJ-II",
        "center": [-0.3, 0.2],
        "r0": 0.25,
        "lam": 1.0,
        "diffusion_amp": 0.3,
        "GJ": {"nums": args.quad},
        "MC": {"nums": args.quad, "eps": 1e-8},
    })
    pde = JAXIrregularHoleDW(cfg, weighting_cfg())
    sampler = IrregularHoleSampler(
        cfg.tlim,
        {"in": args.batch_in, "bd": args.batch_bd, "init": args.batch_init},
        center=cfg.center,
        r0=cfg.r0,
        n_devices=jax.local_device_count(),
        seed=args.seed,
        shard=True,
    )
    state, loss = train_smoke(pde, sampler, 3, args.steps, args.seed)

    grid = np.linspace(-1.0, 1.0, args.grid_2d)
    x_grid, y_grid = np.meshgrid(grid, grid, indexing="xy")
    center = np.asarray(cfg.center, dtype=float)
    mask = ((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2) >= cfg.r0 ** 2

    true_grids = []
    pred_grids = []
    times = parse_time_slices(args.time_slices) if args.time_slices else [0.5, 1.0]
    for time_value in times:
        points = np.stack([
            np.full(np.count_nonzero(mask), time_value),
            x_grid[mask],
            y_grid[mask],
        ], axis=1)
        true_vals = pde.exact(points).reshape(-1)
        pred_vals = predict(state.apply_fn, state.params, points)
        true_grid = np.full_like(x_grid, np.nan, dtype=float)
        pred_grid = np.full_like(x_grid, np.nan, dtype=float)
        true_grid[mask] = true_vals
        pred_grid[mask] = pred_vals
        true_grids.append(true_grid)
        pred_grids.append(pred_grid)

    paths, rel_err = plot_2d_time_slice_files(
        "irregular_hole",
        x_grid,
        y_grid,
        true_grids,
        pred_grids,
        times,
        args.outdir,
    )
    return {
        "case": "irregular_hole",
        "path": ";".join(str(path) for path in paths),
        "relative_error": rel_err,
        "loss": loss,
    }


def lshape_reference_slices(args):
    data = np.load(args.lshape_data)
    times = data["times"]
    x_grid = data["x_grid"]
    y_grid = data["y_grid"]
    snapshots = data["snapshots"]
    meta = {
        "alpha": float(data["alpha"]) if "alpha" in data else 1.5,
        "diffusion_scale": (
            float(data["diffusion_scale"]) if "diffusion_scale" in data else 1.0
        ),
        "t_final": float(data["t_final"]) if "t_final" in data else float(times[-1]),
    }
    requested_times = (
        parse_time_slices(args.time_slices)
        if args.time_slices
        else [0.5 * meta["t_final"], meta["t_final"]]
    )
    selected = []
    for requested in requested_times:
        idx = int(np.argmin(np.abs(times - requested)))
        selected.append((float(times[idx]), snapshots[idx]))
    return x_grid, y_grid, selected, meta


def make_lshape(args):
    x_grid, y_grid, selected, meta = lshape_reference_slices(args)
    cfg = OmegaConf.create({
        "al": meta["alpha"],
        "tlim": [0, meta["t_final"]],
        "method": "GJ-II",
        "diffusion_scale": meta["diffusion_scale"],
        "velocity_scale": 0.2,
        "GJ": {"nums": args.quad},
        "MC": {"nums": args.quad, "eps": 1e-8},
    })
    pde = JAXLShapeDW(cfg, weighting_cfg())
    sampler = LShapeSampler(
        cfg.tlim,
        {"in": args.batch_in, "bd": args.batch_bd, "init": args.batch_init},
        n_devices=jax.local_device_count(),
        seed=args.seed,
        shard=True,
    )
    state, loss = train_smoke(pde, sampler, 3, args.steps, args.seed)

    true_grids = []
    pred_grids = []
    times = []
    for time_value, true_grid in selected:
        mask = np.isfinite(true_grid)
        points = np.stack([
            np.full(np.count_nonzero(mask), time_value),
            x_grid[mask],
            y_grid[mask],
        ], axis=1)
        pred_vals = predict(state.apply_fn, state.params, points)
        pred_grid = np.full_like(true_grid, np.nan, dtype=float)
        pred_grid[mask] = pred_vals
        true_grids.append(true_grid)
        pred_grids.append(pred_grid)
        times.append(time_value)

    paths, rel_err = plot_2d_time_slice_files(
        "lshape",
        x_grid,
        y_grid,
        true_grids,
        pred_grids,
        times,
        args.outdir,
    )
    return {
        "case": "lshape",
        "path": ";".join(str(path) for path in paths),
        "relative_error": rel_err,
        "loss": loss,
    }


def write_summary(outdir, row):
    path = outdir / "summary.csv"
    rows = []
    if path.exists():
        with path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if r["case"] != row["case"]]
    rows.append({
        "case": row["case"],
        "relative_error": f"{row['relative_error']:.8e}",
        "final_smoke_loss": f"{row['loss']:.8e}",
        "figure": row["path"],
    })
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "relative_error", "final_smoke_loss", "figure"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["burgers", "forward", "irregular_hole", "lshape"],
                        required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--quad", type=int, default=3)
    parser.add_argument("--batch-in", type=int, default=8)
    parser.add_argument("--batch-bd", type=int, default=4)
    parser.add_argument("--batch-init", type=int, default=4)
    parser.add_argument("--grid-1d", type=int, default=80)
    parser.add_argument("--grid-2d", type=int, default=90)
    parser.add_argument("--time-slices", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/smoke_results"))
    parser.add_argument("--burgers-data", type=Path, default=Path("data/burgers_150.npz"))
    parser.add_argument("--lshape-data", type=Path,
                        default=Path("data/lshape/lshape_reference.npz"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    makers = {
        "burgers": make_burgers,
        "forward": make_forward,
        "irregular_hole": make_irregular_hole,
        "lshape": make_lshape,
    }
    row = makers[args.case](args)
    summary = write_summary(args.outdir, row)
    print(f"case={row['case']}")
    print(f"relative_error={row['relative_error']:.8e}")
    print(f"final_smoke_loss={row['loss']:.8e}")
    print(f"figure={row['path']}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
