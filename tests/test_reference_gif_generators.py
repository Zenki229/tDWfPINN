import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reference_solvers.burgers_1d.generate_burgers_reference_gifs import (
    orient_burgers_solution,
)
from reference_solvers.forward_1d.generate_forward_reference_gif import forward_solution
from reference_solvers.irregular_hole_2d.generate_irregular_hole_reference import (
    HoleConfig,
    exact_solution,
)


def test_burgers_orientation_uses_initial_condition_for_square_data():
    x = np.linspace(-1.0, 1.0, 5)
    t = np.linspace(0.0, 1.0, 5)
    u_tx = np.vstack([-np.sin(np.pi * x) + 0.1 * ti for ti in t])

    orientation, oriented, init_error = orient_burgers_solution(u_tx, x, t)
    assert orientation == "tx"
    assert init_error < 1e-12
    np.testing.assert_allclose(oriented, u_tx)

    orientation, oriented, init_error = orient_burgers_solution(u_tx.T, x, t)
    assert orientation == "xt"
    assert init_error < 1e-12
    np.testing.assert_allclose(oriented, u_tx)


def test_forward_solution_satisfies_boundary_and_initial_profile():
    t = np.array([0.0, 0.5])
    x = np.linspace(0.0, 1.0, 11)
    u = forward_solution(alpha=1.75, lam=1.0, k=1, a=1.0, b=-0.5, t=t, x=x)
    np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(u[:, -1], 0.0, atol=1e-12)
    np.testing.assert_allclose(u[0], np.sin(np.pi * x), atol=1e-12)


def test_irregular_hole_exact_solution_has_zero_initial_state():
    cfg = HoleConfig()
    x = np.array([-0.5, 0.25])
    y = np.array([0.1, -0.4])
    np.testing.assert_allclose(exact_solution(0.0, x, y, cfg), 0.0)
    np.testing.assert_allclose(exact_solution(1.0, x, y, cfg), 0.0)
