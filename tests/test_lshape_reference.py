import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reference_solvers.lshape_2d.generate_lshape_reference import (
    frac_bdf_coeff,
    solve_modal_backward_euler,
)


def test_frac_bdf_coeff_matches_backward_euler_recurrence():
    weights = frac_bdf_coeff(4, 1.5)
    np.testing.assert_allclose(weights, np.array([1.0, -1.5, 0.375, 0.0625]))


def test_modal_backward_euler_keeps_linear_null_mode():
    times, modal_history, _ = solve_modal_backward_euler(
        alpha=1.5,
        lambdas=np.array([0.0]),
        coeff_g=np.array([2.0]),
        coeff_gt=np.array([-0.5]),
        t_final=1.0,
        n_steps=8,
    )

    expected = 2.0 - 0.5 * times
    np.testing.assert_allclose(modal_history[:, 0], expected, atol=1e-12)
