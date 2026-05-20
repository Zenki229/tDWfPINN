#!/usr/bin/env python3
r"""
Self-contained finite-difference eigenmode reference generator for an L-shaped
fractional diffusion-wave benchmark.

This script is intended for visual preview and for producing reference figures
and animations.  It does not require scikit-fem.  For a higher-fidelity FEM
version, see lshape_fractional_dw_fem_reference.py in the same package.

Model:
    D_t^alpha u - Delta u = 0,                  (t, x, y) in (0, T] x Omega_L,
    u = 0,                                      on partial Omega_L,
    u(0, x, y) = g(x, y),   u_t(0, x, y) = 0.2 g(x, y),

where Omega_L = [-1, 1]^2 \ [0, 1]^2 is the standard L-shaped domain.

The reference solution is computed using discrete Laplace eigenmodes:
    u_ref(t) = sum_j c_j E_{alpha,1}(-lambda_j t^alpha) phi_j
             + sum_j d_j t E_{alpha,2}(-lambda_j t^alpha) phi_j.

Outputs:
    figures/lshape_fd_reference_profile_t05.png
    figures/lshape_fd_reference_surface_t05.png
    figures/lshape_fd_reference_evolution_slice_xminus05.png
    figures/lshape_heatmap_evolution.gif
    figures/lshape_heatmap_evolution_preview.png

Dependencies:
    numpy scipy matplotlib mpmath imageio
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    imageio = None


def build_lshape_fd(N: int = 128):
    """Build a five-point finite-difference Laplacian on an L-shaped domain."""
    xs = np.linspace(-1.0, 1.0, N)
    ys = np.linspace(-1.0, 1.0, N)
    h = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    tol = 1e-12
    outer_boundary = (np.abs(X) >= 1.0 - tol) | (np.abs(Y) >= 1.0 - tol)
    removed_quadrant = (X > tol) & (Y > tol)
    internal_boundary = ((np.abs(X) < tol) & (Y >= -tol)) | ((np.abs(Y) < tol) & (X >= -tol))

    interior = (~outer_boundary) & (~removed_quadrant) & (~internal_boundary)
    visible_domain = ~removed_quadrant

    idx = -np.ones((N, N), dtype=int)
    pts = np.argwhere(interior)  # rows are y-index, x-index
    for k, (j, i) in enumerate(pts):
        idx[j, i] = k

    rows, cols, data = [], [], []
    invh2 = 1.0 / h**2
    for k, (j, i) in enumerate(pts):
        rows.append(k)
        cols.append(k)
        data.append(4.0 * invh2)
        for dj, di in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            jj, ii = j + dj, i + di
            if 0 <= jj < N and 0 <= ii < N:
                kk = idx[jj, ii]
                if kk >= 0:
                    rows.append(k)
                    cols.append(kk)
                    data.append(-invh2)

    K = sp.csr_matrix((data, (rows, cols)), shape=(len(pts), len(pts)))
    return xs, ys, X, Y, visible_domain, pts, K


def mittag_leffler(z: float, alpha: float, beta: float = 1.0, tol: float = 1e-12, max_terms: int = 800) -> float:
    """Series evaluation of E_{alpha,beta}(z), adequate for this visual reference."""
    mp.mp.dps = 50
    z_mp = mp.mpf(z)
    alpha_mp = mp.mpf(alpha)
    beta_mp = mp.mpf(beta)
    s = mp.mpf("0.0")
    zpow = mp.mpf("1.0")
    for k in range(max_terms):
        term = zpow / mp.gamma(alpha_mp * k + beta_mp)
        s_new = s + term
        if abs(term) < tol * max(1.0, abs(s_new)):
            return float(s_new)
        s = s_new
        zpow *= z_mp
    return float(s)


def make_reference(alpha: float = 1.55, N: int = 128, nmodes: int = 36):
    """Compute eigenmodes and modal coefficients for the reference solution."""
    xs, ys, X, Y, visible_domain, pts, K = build_lshape_fd(N)
    eigs, modes = spla.eigsh(K, k=nmodes, which="SM", tol=1e-7)
    order = np.argsort(eigs)
    eigs = eigs[order]
    modes = modes[:, order]

    coords = np.array([(xs[i], ys[j]) for j, i in pts])
    xv, yv = coords[:, 0], coords[:, 1]

    # Smooth sign-changing pulses placed in the L-shaped domain.
    g = (
        np.exp(-((xv + 0.55) ** 2 + (yv + 0.45) ** 2) / 0.08)
        - 0.85 * np.exp(-((xv + 0.55) ** 2 + (yv - 0.45) ** 2) / 0.06)
        + 0.60 * np.exp(-((xv - 0.45) ** 2 + (yv + 0.55) ** 2) / 0.06)
    )
    g_t = 0.20 * g

    coeff_g = modes.T @ g
    coeff_gt = modes.T @ g_t

    def modal_factors(t: float) -> np.ndarray:
        f1 = np.array([mittag_leffler(-lam * t**alpha, alpha, beta=1.0) for lam in eigs])
        f2 = np.array([mittag_leffler(-lam * t**alpha, alpha, beta=2.0) for lam in eigs])
        return coeff_g * f1 + coeff_gt * t * f2

    def solution_grid(t: float) -> np.ndarray:
        U = np.full_like(X, np.nan, dtype=float)
        U[visible_domain] = 0.0
        uvec = modes @ modal_factors(t)
        for val, (j, i) in zip(uvec, pts):
            U[j, i] = val
        return U

    return xs, ys, X, Y, visible_domain, eigs, modes, coeff_g, coeff_gt, solution_grid


def generate_outputs(outdir: Path = Path("figures"), alpha: float = 1.55, N: int = 128, nmodes: int = 36):
    outdir.mkdir(parents=True, exist_ok=True)
    xs, ys, X, Y, visible_domain, eigs, modes, coeff_g, coeff_gt, solution_grid = make_reference(alpha, N, nmodes)

    # Snapshot heatmap at t = 0.5.
    U05 = solution_grid(0.5)
    fig1 = plt.figure(figsize=(7, 6))
    ax1 = fig1.add_subplot(111)
    pcm = ax1.pcolormesh(X, Y, U05, shading="auto")
    ax1.set_title(r"L-shaped numerical reference solution at $t=0.5$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal", adjustable="box")
    fig1.colorbar(pcm, ax=ax1, label=r"$u_{\rm ref}(0.5,x,y)$")
    fig1.tight_layout()
    fig1.savefig(outdir / "lshape_fd_reference_profile_t05.png", dpi=220)
    plt.close(fig1)

    # 3D surface over the L-shaped domain at t = 0.5.
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(X, Y, U05, linewidth=0, antialiased=True)
    ax2.set_title(r"3D surface of the L-shaped reference solution at $t=0.5$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel(r"$u_{\rm ref}$")
    fig2.tight_layout()
    fig2.savefig(outdir / "lshape_fd_reference_surface_t05.png", dpi=220)
    plt.close(fig2)

    # Time evolution on vertical slice x = -0.5.
    x_slice = -0.5
    i_slice = int(np.argmin(np.abs(xs - x_slice)))
    x_actual = xs[i_slice]
    t_values_slice = np.linspace(0.0, 1.0, 121)
    Y_slice, T_slice = np.meshgrid(ys, t_values_slice)
    U_slice = np.full_like(Y_slice, np.nan, dtype=float)
    for nt, tt in enumerate(t_values_slice):
        U_slice[nt, :] = solution_grid(tt)[:, i_slice]

    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot_surface(Y_slice, T_slice, U_slice, linewidth=0, antialiased=True)
    ax3.set_title(rf"Time evolution on the slice $x={x_actual:.2f}$")
    ax3.set_xlabel("y")
    ax3.set_ylabel("t")
    ax3.set_zlabel(r"$u_{\rm ref}(t,x,y)$")
    fig3.tight_layout()
    fig3.savefig(outdir / "lshape_fd_reference_evolution_slice_xminus05.png", dpi=220)
    plt.close(fig3)

    # Heatmap GIF over time.
    if imageio is None:
        raise RuntimeError("imageio is required to generate the L-shape GIF outputs")

    t_values = np.linspace(0.0, 1.0, 31)
    frames_data = [solution_grid(t) for t in t_values]
    finite_vals = np.concatenate([U[np.isfinite(U)] for U in frames_data])
    vmin, vmax = float(np.min(finite_vals)), float(np.max(finite_vals))

    frame_paths = []
    for k, (t, U) in enumerate(zip(t_values, frames_data)):
        fig = plt.figure(figsize=(6.5, 6))
        ax = fig.add_subplot(111)
        pcm = ax.pcolormesh(X, Y, U, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(rf"L-shaped reference heatmap, $t={t:.2f}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(pcm, ax=ax, label=r"$u_{\rm ref}(t,x,y)$")
        fig.tight_layout()
        frame_path = outdir / f"lshape_heatmap_frame_{k:03d}.png"
        fig.savefig(frame_path, dpi=160)
        plt.close(fig)
        frame_paths.append(frame_path)

    images = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(outdir / "lshape_heatmap_evolution.gif", images, duration=0.14, loop=0)
    imageio.imwrite(outdir / "lshape_heatmap_evolution_preview.png", images[len(images) // 2])

    print("First five eigenvalues:", np.round(eigs[:5], 6))
    print(f"Outputs saved under: {outdir.resolve()}")


if __name__ == "__main__":
    generate_outputs()
