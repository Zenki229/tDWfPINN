import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from PIL import Image
from pymittagleffler import mittag_leffler


def build_lshape_fd(n_grid):
    xs = np.linspace(-1.0, 1.0, n_grid)
    ys = np.linspace(-1.0, 1.0, n_grid)
    h = xs[1] - xs[0]
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")

    tol = 1e-12
    outer_boundary = (np.abs(x_grid) >= 1.0 - tol) | (np.abs(y_grid) >= 1.0 - tol)
    removed_quadrant = (x_grid > tol) & (y_grid > tol)
    internal_boundary = (
        ((np.abs(x_grid) < tol) & (y_grid >= -tol))
        | ((np.abs(y_grid) < tol) & (x_grid >= -tol))
    )

    interior = (~outer_boundary) & (~removed_quadrant) & (~internal_boundary)
    visible_domain = ~removed_quadrant

    idx = -np.ones((n_grid, n_grid), dtype=int)
    pts = np.argwhere(interior)
    for k, (j, i) in enumerate(pts):
        idx[j, i] = k

    rows, cols, data = [], [], []
    invh2 = 1.0 / h ** 2
    for k, (j, i) in enumerate(pts):
        rows.append(k)
        cols.append(k)
        data.append(4.0 * invh2)
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            jj, ii = j + dj, i + di
            if 0 <= jj < n_grid and 0 <= ii < n_grid:
                kk = idx[jj, ii]
                if kk >= 0:
                    rows.append(k)
                    cols.append(kk)
                    data.append(-invh2)

    stiffness = sp.csr_matrix((data, (rows, cols)), shape=(len(pts), len(pts)))
    return xs, ys, x_grid, y_grid, visible_domain, pts, stiffness


def initial_profile(x, y):
    return (
        np.exp(-((x + 0.55) ** 2 + (y + 0.45) ** 2) / 0.08)
        - 0.85 * np.exp(-((x + 0.55) ** 2 + (y - 0.45) ** 2) / 0.06)
        + 0.60 * np.exp(-((x - 0.45) ** 2 + (y + 0.55) ** 2) / 0.06)
    )


def modal_factors(t_value, alpha, lambdas, coeff_g, coeff_gt):
    z = -lambdas * t_value ** alpha
    f1 = np.real(mittag_leffler(z, alpha, 1.0))
    f2 = np.real(mittag_leffler(z, alpha, 2.0))
    return coeff_g * f1 + coeff_gt * t_value * f2


def solution_vector(t_value, alpha, lambdas, modes, coeff_g, coeff_gt):
    return modes @ modal_factors(t_value, alpha, lambdas, coeff_g, coeff_gt)


def vector_to_grid(u_vec, x_grid, visible_domain, pts):
    u_grid = np.full_like(x_grid, np.nan, dtype=float)
    u_grid[visible_domain] = 0.0
    for val, (j, i) in zip(u_vec, pts):
        u_grid[j, i] = val
    return u_grid


def render_frame(x_grid, y_grid, u_grid, t_value, vmin, vmax):
    fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=120)
    pcm = ax.pcolormesh(x_grid, y_grid, u_grid, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_title(rf"L-shaped reference, $t={t_value:.2f}$")
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


def generate_reference(alpha, n_grid, n_modes, n_frames, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    xs, ys, x_grid, y_grid, visible_domain, pts, stiffness = build_lshape_fd(n_grid)
    lambdas, modes = spla.eigsh(stiffness, k=n_modes, which="SM", tol=1e-7)
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    modes = modes[:, order]

    coords = np.array([(xs[i], ys[j]) for j, i in pts])
    g = initial_profile(coords[:, 0], coords[:, 1])
    g_t = 0.20 * g
    coeff_g = modes.T @ g
    coeff_gt = modes.T @ g_t

    times = np.linspace(0.0, 1.0, n_frames)
    snapshots = []
    for t_value in times:
        u_vec = solution_vector(t_value, alpha, lambdas, modes, coeff_g, coeff_gt)
        snapshots.append(vector_to_grid(u_vec, x_grid, visible_domain, pts))
    snapshots = np.asarray(snapshots)

    np.savez_compressed(
        outdir / "lshape_reference.npz",
        alpha=alpha,
        n_grid=n_grid,
        n_modes=n_modes,
        times=times,
        x=xs,
        y=ys,
        x_grid=x_grid,
        y_grid=y_grid,
        visible_domain=visible_domain,
        interior_indices=pts,
        lambdas=lambdas,
        modes=modes,
        coeff_g=coeff_g,
        coeff_gt=coeff_gt,
        snapshots=snapshots,
    )

    finite_vals = snapshots[np.isfinite(snapshots)]
    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    frames = [
        render_frame(x_grid, y_grid, u_grid, t_value, vmin, vmax)
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
    print("first five eigenvalues:", np.round(lambdas[:5], 6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--n-grid", type=int, default=81)
    parser.add_argument("--n-modes", type=int, default=36)
    parser.add_argument("--n-frames", type=int, default=41)
    parser.add_argument("--outdir", type=Path, default=Path("data/lshape"))
    args = parser.parse_args()
    generate_reference(args.alpha, args.n_grid, args.n_modes, args.n_frames, args.outdir)


if __name__ == "__main__":
    main()
