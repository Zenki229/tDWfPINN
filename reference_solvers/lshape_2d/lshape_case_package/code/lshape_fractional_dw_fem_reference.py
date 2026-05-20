#!/usr/bin/env python3
"""
High-fidelity FEM reference generator for an L-shaped time-fractional
diffusion-wave benchmark.

Recommended use in the revised paper:
    Use this solver to generate a numerical reference solution on an
    irregular L-shaped domain. The PINN is trained on the strong-form PDE,
    while its accuracy is evaluated against this high-resolution FEM reference.

PDE:
    D_t^alpha u - div(a(x,y) grad u) = 0,      (t,x,y) in (0,T] x Omega_L
    u = 0,                                    on boundary
    u(0,x,y) = g_h(x,y),   u_t(0,x,y) = 0,

where Omega_L = [-1,1]^2 \ [0,1]^2 is the standard L-shaped domain.

The reference is constructed by the generalized FEM eigenproblem
    K_a phi_j = lambda_j M phi_j,
and the modal solution
    u_ref(t) = sum_j c_j E_{alpha,1}(-lambda_j t^alpha) phi_j.

Dependencies:
    pip install scikit-fem scipy matplotlib mpmath

Notes:
    * This is a reference generator, not part of the proposed PINN algorithm.
    * For production-level runs, increase nrefs and nmodes.
    * For large eigenvalues, a specialized Mittag-Leffler package is preferable
      to the simple series evaluation below.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, spsolve
import mpmath as mp

from skfem import MeshTri, Basis, ElementTriP2, BilinearForm, asm
from skfem.models.poisson import mass
from skfem.helpers import dot, grad


def a_coeff(x, y):
    """Positive heterogeneous diffusion coefficient."""
    return 1.0 + 0.25 * np.sin(np.pi * x) * np.cos(np.pi * y)


@BilinearForm
def diffusion(u, v, w):
    x, y = w.x
    return a_coeff(x, y) * dot(grad(u), grad(v))


def mittag_leffler_E_alpha_1(z, alpha, tol=1e-13, max_terms=1000):
    """Evaluate E_{alpha,1}(z) by the defining series.

    For the low number of modes used for visualization, this is adequate.
    For large |z| or many modes, replace by a dedicated implementation.
    """
    z = mp.mpf(z)
    alpha = mp.mpf(alpha)
    s = mp.mpf("0.0")
    zpow = mp.mpf("1.0")
    for k in range(max_terms):
        term = zpow / mp.gamma(alpha * k + 1)
        s_new = s + term
        if abs(term) < tol * max(1.0, abs(s_new)):
            return float(s_new)
        s = s_new
        zpow *= z
    return float(s)


def build_lshape_mesh(nrefs=6):
    # Standard L-shaped domain from scikit-fem:
    # vertices cover [-1,1]^2 with the upper-right quadrant removed.
    return MeshTri.init_lshaped().refined(nrefs).smoothed()


def solve_eigenbasis(mesh, nmodes=25):
    basis = Basis(mesh, ElementTriP2())
    K = asm(diffusion, basis)
    M = asm(mass, basis)

    # Homogeneous Dirichlet boundary: keep only interior degrees of freedom.
    D = basis.get_dofs().all()
    all_dofs = np.arange(basis.N)
    I = np.setdiff1d(all_dofs, D)

    Kc = K[I][:, I]
    Mc = M[I][:, I]

    vals, vecs = eigsh(Kc, k=nmodes, M=Mc, sigma=0.0, which="LM")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    modes = np.zeros((basis.N, nmodes))
    for j in range(nmodes):
        full = np.zeros(basis.N)
        full[I] = vecs[:, j]
        norm = np.sqrt(full @ (M @ full))
        modes[:, j] = full / norm

    return basis, M, vals, modes


def initial_profile_at_dofs(basis):
    """A smooth localized initial pulse; boundary dofs will be set to zero."""
    x, y = basis.doflocs
    g1 = np.exp(-((x + 0.45) ** 2 + (y + 0.35) ** 2) / 0.12)
    g2 = 0.65 * np.exp(-((x + 0.20) ** 2 + (y - 0.55) ** 2) / 0.08)
    g = g1 + g2

    # Enforce homogeneous Dirichlet boundary in the FE vector.
    D = basis.get_dofs().all()
    g[D] = 0.0
    return g


def modal_coefficients(M, modes, g_vec):
    """c_j = <g_h, phi_j>_M because phi_j are M-orthonormal."""
    return modes.T @ (M @ g_vec)


def modal_time_coeffs(t, alpha, lambdas, coeffs):
    out = np.zeros_like(coeffs)
    for j, lam in enumerate(lambdas):
        out[j] = coeffs[j] * mittag_leffler_E_alpha_1(-lam * t ** alpha, alpha)
    return out


def reference_vector(t, alpha, lambdas, coeffs, modes):
    return modes @ modal_time_coeffs(t, alpha, lambdas, coeffs)


def plot_snapshot(mesh, uh, title, filename):
    from skfem.visuals.matplotlib import draw, plot

    fig, ax = plt.subplots(figsize=(7, 6))
    draw(mesh, ax=ax, color="0.7", linewidth=0.15)
    plot(mesh, uh, ax=ax, shading="gouraud", colorbar=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(filename, dpi=220)
    print(f"saved: {filename}")


def main():
    alpha = 1.5
    mesh = build_lshape_mesh(nrefs=6)
    basis, M, lambdas, modes = solve_eigenbasis(mesh, nmodes=25)

    print("First FEM eigenvalues of -div(a grad) on the L-shaped domain:")
    for j, lam in enumerate(lambdas[:10], start=1):
        print(f"  lambda_{j:02d} = {lam:.10f}")

    g = initial_profile_at_dofs(basis)
    coeffs = modal_coefficients(M, modes, g)

    for t in [0.25, 0.50, 0.75]:
        uh = reference_vector(t, alpha, lambdas, coeffs, modes)
        plot_snapshot(
            mesh,
            uh,
            title=rf"L-shaped FEM reference, $\alpha={alpha}$, $t={t}$",
            filename=f"lshape_fem_reference_alpha{alpha}_t{t:.2f}.png",
        )

    # Save reference data for PINN test-set interpolation or plotting.
    np.savez(
        "lshape_fem_reference_data.npz",
        p=mesh.p,
        t=mesh.t,
        lambdas=lambdas,
        coeffs=coeffs,
        modes=modes,
        alpha=alpha,
    )


if __name__ == "__main__":
    main()
