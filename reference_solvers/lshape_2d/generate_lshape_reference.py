import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lshape_case_package.code.lshape_fd_reference_and_gif import build_lshape_fd


def initial_profile(x, y):
    return (
        np.exp(-((x + 0.55) ** 2 + (y + 0.45) ** 2) / 0.08)
        - 0.85 * np.exp(-((x + 0.55) ** 2 + (y - 0.45) ** 2) / 0.06)
        + 0.60 * np.exp(-((x - 0.45) ** 2 + (y + 0.55) ** 2) / 0.06)
    )


def frac_bdf_coeff(n_coeffs, alpha):
    """Backward-Euler CQ weights for (1 - z)^alpha.

    This is the same recurrence used by reference_solvers/burgers_1d/frac_utils.py.
    """
    weights = np.zeros(n_coeffs, dtype=float)
    weights[0] = 1.0
    for j in range(n_coeffs - 1):
        weights[j + 1] = -weights[j] * (alpha - j) / (j + 1)
    return weights


def solve_modal_backward_euler(alpha, lambdas, coeff_g, coeff_gt, t_final, n_steps):
    """Solve D_t^alpha q_j + lambda_j q_j = 0 by BE convolution quadrature.

    The Caputo correction subtracts q_j(0) + t q'_j(0), matching the
    history-correction form in reference_solvers/burgers_1d/frac_weno_solver.py.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if not 1.0 < alpha < 2.0:
        raise ValueError("alpha must be in (1, 2) for the diffusion-wave case")

    times = np.linspace(0.0, t_final, n_steps + 1)
    dt = times[1] - times[0]
    weights = frac_bdf_coeff(n_steps + 1, alpha)

    coeffs = np.zeros((n_steps + 1, len(lambdas)), dtype=float)
    coeffs[0] = coeff_g
    denom = weights[0] + (dt ** alpha) * lambdas

    for n in range(1, n_steps + 1):
        history = weights[1 : n + 1] @ coeffs[n - 1 :: -1]
        displacement_correction = np.sum(weights[: n + 1]) * coeff_g
        velocity_correction = np.dot(weights[: n + 1], times[n::-1]) * coeff_gt
        coeffs[n] = (displacement_correction + velocity_correction - history) / denom

    return times, coeffs, weights


def sample_modal_coefficients(solver_times, modal_coeffs, frame_times):
    sampled = np.empty((len(frame_times), modal_coeffs.shape[1]), dtype=float)
    for j in range(modal_coeffs.shape[1]):
        sampled[:, j] = np.interp(frame_times, solver_times, modal_coeffs[:, j])
    return sampled


def vector_to_grid(u_vec, x_grid, visible_domain, pts):
    u_grid = np.full_like(x_grid, np.nan, dtype=float)
    u_grid[visible_domain] = 0.0
    for val, (j, i) in zip(u_vec, pts):
        u_grid[j, i] = val
    return u_grid


def render_frame(x_grid, y_grid, u_grid, t_value, vmin, vmax, alpha, diffusion_scale):
    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=120)
    pcm = ax.pcolormesh(
        x_grid,
        y_grid,
        u_grid,
        shading="auto",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        rf"L-shaped reference, $\alpha={alpha:.2f}$, "
        rf"$\kappa={diffusion_scale:.2f}$, $t={t_value:.2f}$"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(pcm, ax=ax, label=r"$u_{\rm ref}(t,x,y)$")
    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    image = Image.fromarray(frame[:, :, :3])
    plt.close(fig)
    return image


def generate_reference(
    alpha,
    diffusion_scale,
    n_grid,
    n_modes,
    n_frames,
    n_steps,
    t_final,
    outdir,
):
    outdir.mkdir(parents=True, exist_ok=True)

    xs, ys, x_grid, y_grid, visible_domain, pts, stiffness = build_lshape_fd(n_grid)
    lambdas, modes = spla.eigsh(stiffness, k=n_modes, which="SM", tol=1e-7)
    order = np.argsort(lambdas)
    laplacian_lambdas = lambdas[order]
    lambdas = diffusion_scale * laplacian_lambdas
    modes = modes[:, order]

    coords = np.array([(xs[i], ys[j]) for j, i in pts])
    g = initial_profile(coords[:, 0], coords[:, 1])
    g_t = 0.20 * g
    coeff_g = modes.T @ g
    coeff_gt = modes.T @ g_t

    solver_times, modal_history, bdf_weights = solve_modal_backward_euler(
        alpha,
        lambdas,
        coeff_g,
        coeff_gt,
        t_final,
        n_steps,
    )
    times = np.linspace(0.0, t_final, n_frames)
    frame_modal_coeffs = sample_modal_coefficients(solver_times, modal_history, times)
    solution_vectors = frame_modal_coeffs @ modes.T

    snapshots = []
    for u_vec in solution_vectors:
        snapshots.append(vector_to_grid(u_vec, x_grid, visible_domain, pts))
    snapshots = np.asarray(snapshots)

    np.savez_compressed(
        outdir / "lshape_reference.npz",
        alpha=alpha,
        diffusion_scale=diffusion_scale,
        n_grid=n_grid,
        n_modes=n_modes,
        n_steps=n_steps,
        t_final=t_final,
        time_step=solver_times[1] - solver_times[0],
        time_method="backward_euler_convolution_quadrature",
        times=times,
        solver_times=solver_times,
        x=xs,
        y=ys,
        x_grid=x_grid,
        y_grid=y_grid,
        visible_domain=visible_domain,
        interior_indices=pts,
        laplacian_lambdas=laplacian_lambdas,
        lambdas=lambdas,
        modes=modes,
        coeff_g=coeff_g,
        coeff_gt=coeff_gt,
        bdf_weights=bdf_weights,
        modal_history=modal_history,
        frame_modal_coeffs=frame_modal_coeffs,
        snapshots=snapshots,
    )

    finite_vals = snapshots[np.isfinite(snapshots)]
    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    frames = [
        render_frame(x_grid, y_grid, u_grid, t_value, vmin, vmax, alpha, diffusion_scale)
        for t_value, u_grid in zip(times, snapshots)
    ]
    frames[0].save(
        outdir / "lshape_reference.gif",
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
    )

    print(f"saved: {outdir / 'lshape_reference.npz'}")
    print(f"saved: {outdir / 'lshape_reference.gif'}")
    print(
        "time method: backward Euler convolution quadrature, "
        f"n_steps={n_steps}, dt={solver_times[1] - solver_times[0]:.6g}, "
        f"diffusion_scale={diffusion_scale}"
    )
    print("first five eigenvalues:", np.round(lambdas[:5], 6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.8)
    parser.add_argument("--diffusion-scale", type=float, default=0.25)
    parser.add_argument("--n-grid", type=int, default=128)
    parser.add_argument("--n-modes", type=int, default=36)
    parser.add_argument("--n-frames", type=int, default=81)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--t-final", type=float, default=5.0)
    parser.add_argument("--outdir", type=Path, default=Path("data/lshape"))
    args = parser.parse_args()
    generate_reference(
        args.alpha,
        args.diffusion_scale,
        args.n_grid,
        args.n_modes,
        args.n_frames,
        args.n_steps,
        args.t_final,
        args.outdir,
    )


if __name__ == "__main__":
    main()
