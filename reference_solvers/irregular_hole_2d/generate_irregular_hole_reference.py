import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from PIL import Image
from scipy.special import gamma as sp_gamma


@dataclass(frozen=True)
class HoleConfig:
    alpha: float = 1.5
    center: tuple[float, float] = (-0.3, 0.2)
    radius: float = 0.25
    diffusion_amp: float = 0.3
    lam: float = 1.0
    t_final: float = 1.0


def float_token(value):
    return f"{float(value):.2f}".replace(".", "p")


def frac_bdf_coeff(n_coeffs, alpha):
    weights = np.zeros(n_coeffs, dtype=float)
    weights[0] = 1.0
    for j in range(n_coeffs - 1):
        weights[j + 1] = -weights[j] * (alpha - j) / (j + 1)
    return weights


def q_time(t):
    return t**2 * (1.0 - t) ** 2


def dtalpha_q(t, alpha):
    safe_t = np.maximum(t, 1e-14)
    return (
        2.0 / sp_gamma(3.0 - alpha) * safe_t ** (2.0 - alpha)
        - 12.0 / sp_gamma(4.0 - alpha) * safe_t ** (3.0 - alpha)
        + 24.0 / sp_gamma(5.0 - alpha) * safe_t ** (4.0 - alpha)
    )


def phi_terms(x, y, cfg):
    cx, cy = cfg.center
    psi = (1.0 - x**2) * (1.0 - y**2)
    chi = (x - cx) ** 2 + (y - cy) ** 2 - cfg.radius**2
    phi = psi * chi

    psi_x = -2.0 * x * (1.0 - y**2)
    psi_y = -2.0 * y * (1.0 - x**2)
    lap_psi = 2.0 * x**2 + 2.0 * y**2 - 4.0

    chi_x = 2.0 * (x - cx)
    chi_y = 2.0 * (y - cy)
    lap_chi = 4.0

    phi_x = chi * psi_x + psi * chi_x
    phi_y = chi * psi_y + psi * chi_y
    lap_phi = chi * lap_psi + 2.0 * (psi_x * chi_x + psi_y * chi_y) + psi * lap_chi
    return phi, phi_x, phi_y, lap_phi


def diffusion_coeff(x, y, cfg):
    return 1.0 + cfg.diffusion_amp * np.sin(np.pi * x) * np.cos(np.pi * y)


def grad_diffusion_coeff(x, y, cfg):
    ax = cfg.diffusion_amp * np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
    ay = -cfg.diffusion_amp * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)
    return ax, ay


def source_values(t, x, y, cfg):
    phi, phi_x, phi_y, lap_phi = phi_terms(x, y, cfg)
    q = q_time(t)
    dtalpha = dtalpha_q(t, cfg.alpha)
    a = diffusion_coeff(x, y, cfg)
    ax, ay = grad_diffusion_coeff(x, y, cfg)
    bx = 1.0 + y
    by = x - 1.0
    return (
        dtalpha * phi
        - q * (ax * phi_x + ay * phi_y + a * lap_phi)
        + q * (bx * phi_x + by * phi_y)
        + cfg.lam * (q**3) * (phi**3)
    )


def exact_solution(t, x, y, cfg):
    phi, _, _, _ = phi_terms(x, y, cfg)
    return q_time(t) * phi


def build_hole_grid(n_grid, cfg):
    xs = np.linspace(-1.0, 1.0, n_grid)
    ys = np.linspace(-1.0, 1.0, n_grid)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    h = xs[1] - xs[0]

    cx, cy = cfg.center
    hole = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 < cfg.radius**2
    outer_boundary = (np.isclose(np.abs(x_grid), 1.0)) | (np.isclose(np.abs(y_grid), 1.0))
    domain = ~hole
    unknown = domain & ~outer_boundary

    index = -np.ones((n_grid, n_grid), dtype=int)
    pts = np.argwhere(unknown)
    for k, (j, i) in enumerate(pts):
        index[j, i] = k
    coords = np.array([(xs[i], ys[j]) for j, i in pts], dtype=float)
    return xs, ys, x_grid, y_grid, domain, unknown, pts, index, coords, h


def add_neighbor(rows, cols, data, row, index, jj, ii, value):
    if 0 <= jj < index.shape[0] and 0 <= ii < index.shape[1]:
        col = index[jj, ii]
        if col >= 0:
            rows.append(row)
            cols.append(col)
            data.append(value)


def build_operator(xs, ys, pts, index, h, cfg):
    rows, cols, data = [], [], []
    inv_h = 1.0 / h
    inv_h2 = inv_h * inv_h

    for row, (j, i) in enumerate(pts):
        x = xs[i]
        y = ys[j]
        diag = 0.0

        faces = [
            (j, i + 1, diffusion_coeff(x + 0.5 * h, y, cfg)),
            (j, i - 1, diffusion_coeff(x - 0.5 * h, y, cfg)),
            (j + 1, i, diffusion_coeff(x, y + 0.5 * h, cfg)),
            (j - 1, i, diffusion_coeff(x, y - 0.5 * h, cfg)),
        ]
        for jj, ii, face_a in faces:
            coeff = face_a * inv_h2
            diag += coeff
            add_neighbor(rows, cols, data, row, index, jj, ii, -coeff)

        bx = 1.0 + y
        by = x - 1.0
        if bx >= 0.0:
            diag += bx * inv_h
            add_neighbor(rows, cols, data, row, index, j, i - 1, -bx * inv_h)
        else:
            diag += -bx * inv_h
            add_neighbor(rows, cols, data, row, index, j, i + 1, bx * inv_h)

        if by >= 0.0:
            diag += by * inv_h
            add_neighbor(rows, cols, data, row, index, j - 1, i, -by * inv_h)
        else:
            diag += -by * inv_h
            add_neighbor(rows, cols, data, row, index, j + 1, i, by * inv_h)

        rows.append(row)
        cols.append(row)
        data.append(diag)

    return sp.csr_matrix((data, (rows, cols)), shape=(len(pts), len(pts)))


def vector_to_grid(values, shape, pts, domain):
    grid = np.full(shape, np.nan, dtype=float)
    grid[domain] = 0.0
    for val, (j, i) in zip(values, pts):
        grid[j, i] = val
    return grid


def render_frame(
    x_grid,
    y_grid,
    values,
    t_value,
    vmin,
    vmax,
    cfg,
    title_prefix,
    colorbar_label,
    error_text=None,
):
    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=120)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")
    mesh = ax.pcolormesh(
        x_grid,
        y_grid,
        values,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    circle = plt.Circle(cfg.center, cfg.radius, color="black", fill=False, lw=1.4)
    ax.add_patch(circle)
    title = rf"{title_prefix}, $\alpha={cfg.alpha:.2f}$, $t={t_value:.3f}$"
    if error_text:
        title += f", {error_text}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    image = Image.fromarray(frame[:, :, :3])
    plt.close(fig)
    return image


def solve_reference(args):
    cfg = HoleConfig(
        alpha=args.alpha,
        center=(args.center_x, args.center_y),
        radius=args.radius,
        diffusion_amp=args.diffusion_amp,
        lam=args.lam,
        t_final=args.t_final,
    )
    (
        xs,
        ys,
        x_grid,
        y_grid,
        domain,
        unknown,
        pts,
        index,
        coords,
        h,
    ) = build_hole_grid(args.n_grid, cfg)
    operator = build_operator(xs, ys, pts, index, h, cfg)

    times = np.linspace(0.0, cfg.t_final, args.n_steps + 1)
    dt = times[1] - times[0]
    weights = frac_bdf_coeff(args.n_steps + 1, cfg.alpha)
    dt_alpha = dt ** (-cfg.alpha)
    system = (dt_alpha * sp.eye(operator.shape[0], format="csr") + operator).tocsc()
    solver = spla.factorized(system)

    history = np.zeros((args.n_steps + 1, operator.shape[0]), dtype=float)
    frame_indices = np.unique(
        np.linspace(0, args.n_steps, min(args.n_frames, args.n_steps + 1), dtype=int)
    )
    frame_set = set(int(i) for i in frame_indices)
    snapshots = []
    exact_snapshots = []
    frame_times = []
    frame_rel_errors = []
    frame_abs_errors = []
    frame_exact_norms = []

    x = coords[:, 0]
    y = coords[:, 1]
    zero_exact = exact_solution(0.0, x, y, cfg)
    if 0 in frame_set:
        snapshots.append(vector_to_grid(history[0], x_grid.shape, pts, domain))
        exact_snapshots.append(vector_to_grid(zero_exact, x_grid.shape, pts, domain))
        frame_times.append(0.0)
        frame_rel_errors.append(0.0)
        frame_abs_errors.append(0.0)
        frame_exact_norms.append(0.0)

    for n in range(1, args.n_steps + 1):
        t_value = times[n]
        conv_history = weights[1 : n + 1] @ history[n - 1 :: -1]
        rhs = (
            source_values(t_value, x, y, cfg)
            - dt_alpha * conv_history
            - cfg.lam * history[n - 1] ** 3
        )
        history[n] = solver(rhs)

        if n in frame_set:
            exact = exact_solution(t_value, x, y, cfg)
            abs_err = np.linalg.norm(history[n] - exact)
            exact_norm = np.linalg.norm(exact)
            snapshots.append(vector_to_grid(history[n], x_grid.shape, pts, domain))
            exact_snapshots.append(vector_to_grid(exact, x_grid.shape, pts, domain))
            frame_times.append(float(t_value))
            frame_abs_errors.append(float(abs_err))
            frame_exact_norms.append(float(exact_norm))

    frame_abs_errors = np.asarray(frame_abs_errors)
    frame_exact_norms = np.asarray(frame_exact_norms)
    error_scale = float(np.max(frame_exact_norms))
    frame_rel_errors = frame_abs_errors / (error_scale + 1e-14)

    return {
        "config": cfg,
        "x": xs,
        "y": ys,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "domain": domain,
        "unknown": unknown,
        "interior_indices": pts,
        "times": np.asarray(frame_times),
        "solver_times": times,
        "snapshots": np.asarray(snapshots),
        "exact_snapshots": np.asarray(exact_snapshots),
        "frame_indices": frame_indices,
        "frame_rel_errors": frame_rel_errors,
        "frame_abs_errors": frame_abs_errors,
        "frame_exact_norms": frame_exact_norms,
        "error_scale": error_scale,
        "final_vector": history[-1],
        "bdf_weights": weights,
        "time_step": dt,
        "h": h,
        "operator_nnz": operator.nnz,
    }


def save_outputs(result, outdir, duration, output_prefix="irregular_hole_reference"):
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = result["config"]
    snapshots = result["snapshots"]
    exact_snapshots = result["exact_snapshots"]
    finite_vals = np.concatenate(
        [
            snapshots[np.isfinite(snapshots)],
            exact_snapshots[np.isfinite(exact_snapshots)],
        ]
    )
    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    npz_path = outdir / f"{output_prefix}.npz"
    reference_gif_path = outdir / f"{output_prefix}.gif"
    exact_gif_path = (
        outdir / "irregular_hole_exact.gif"
        if output_prefix == "irregular_hole_reference"
        else outdir / f"{output_prefix}_exact.gif"
    )

    np.savez_compressed(
        npz_path,
        alpha=cfg.alpha,
        center=np.asarray(cfg.center),
        radius=cfg.radius,
        diffusion_amp=cfg.diffusion_amp,
        lam=cfg.lam,
        t_final=cfg.t_final,
        n_grid=len(result["x"]),
        n_steps=len(result["solver_times"]) - 1,
        time_step=result["time_step"],
        spatial_step=result["h"],
        time_method="backward_euler_convolution_quadrature",
        space_method="masked_finite_difference",
        nonlinear_treatment="explicit_lagged_cubic",
        operator_nnz=result["operator_nnz"],
        x=result["x"],
        y=result["y"],
        x_grid=result["x_grid"],
        y_grid=result["y_grid"],
        domain=result["domain"],
        unknown=result["unknown"],
        interior_indices=result["interior_indices"],
        times=result["times"],
        solver_times=result["solver_times"],
        snapshots=snapshots,
        exact_snapshots=result["exact_snapshots"],
        frame_indices=result["frame_indices"],
        frame_rel_errors=result["frame_rel_errors"],
        frame_abs_errors=result["frame_abs_errors"],
        frame_exact_norms=result["frame_exact_norms"],
        error_scale=result["error_scale"],
        final_vector=result["final_vector"],
        bdf_weights=result["bdf_weights"],
    )

    reference_frames = [
        render_frame(
            result["x_grid"],
            result["y_grid"],
            snapshot,
            t_value,
            vmin,
            vmax,
            cfg,
            "Circular-hole numerical reference",
            r"$u_{\rm ref}(t,x,y)$",
            error_text=f"scaled err. {rel_err:.2e}",
        )
        for snapshot, t_value, rel_err in zip(
            snapshots,
            result["times"],
            result["frame_rel_errors"],
        )
    ]
    reference_frames[0].save(
        reference_gif_path,
        save_all=True,
        append_images=reference_frames[1:],
        duration=duration,
        loop=0,
    )

    exact_frames = [
        render_frame(
            result["x_grid"],
            result["y_grid"],
            exact_snapshot,
            t_value,
            vmin,
            vmax,
            cfg,
            "Circular-hole analytic solution",
            r"$u^*(t,x,y)$",
        )
        for exact_snapshot, t_value in zip(result["exact_snapshots"], result["times"])
    ]
    exact_frames[0].save(
        exact_gif_path,
        save_all=True,
        append_images=exact_frames[1:],
        duration=duration,
        loop=0,
    )

    print(f"saved: {npz_path}")
    print(f"saved: {reference_gif_path}")
    print(f"saved: {exact_gif_path}")
    print(
        "grid: "
        f"{len(result['x'])} x {len(result['y'])}, "
        f"unknowns={int(np.count_nonzero(result['unknown']))}, "
        f"n_steps={len(result['solver_times']) - 1}, "
        f"dt={result['time_step']:.6g}, "
        f"max_scaled_error={np.max(result['frame_rel_errors']):.3e}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("data/irregular_hole"))
    parser.add_argument("--n-grid", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--n-frames", type=int, default=81)
    parser.add_argument("--duration", type=int, default=130)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--output-prefix", default="irregular_hole_reference")
    parser.add_argument(
        "--tag-alpha",
        action="store_true",
        help="append an alpha token to the output prefix, e.g. alpha1p25",
    )
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--center-x", type=float, default=-0.3)
    parser.add_argument("--center-y", type=float, default=0.2)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--diffusion-amp", type=float, default=0.3)
    parser.add_argument("--lam", type=float, default=1.0)
    args = parser.parse_args()

    result = solve_reference(args)
    output_prefix = args.output_prefix
    if args.tag_alpha:
        output_prefix = f"{output_prefix}_alpha{float_token(args.alpha)}"
    save_outputs(result, args.outdir, args.duration, output_prefix)


if __name__ == "__main__":
    main()
