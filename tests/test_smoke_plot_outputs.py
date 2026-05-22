import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.generate_smoke_results import (
    PAPER_COLORMAP,
    PAPER_SHADING,
    float_token,
    lshape_reference_slices,
    plot_1d_case,
    plot_2d_time_slice_files,
    resolve_burgers_data_path,
    resolve_lshape_data_path,
)


def test_smoke_plots_use_paper_colormap():
    assert PAPER_COLORMAP == "jet"
    assert PAPER_SHADING == "gouraud"


def test_1d_smoke_panels_are_written_separately(tmp_path):
    t_grid, x_grid = np.meshgrid(np.linspace(0.0, 1.0, 4), np.linspace(-1.0, 1.0, 3))
    true = np.sin(t_grid + x_grid)
    pred = true + 0.1

    paths, rel_err = plot_1d_case("demo", t_grid, x_grid, true, pred, tmp_path)

    assert rel_err > 0.0
    assert [path.name for path in paths] == [
        "demo_true_smoke.png",
        "demo_sol_smoke.png",
        "demo_abs_error_smoke.png",
    ]
    assert all(path.exists() for path in paths)


def test_2d_smoke_time_slice_panels_are_written_separately(tmp_path):
    x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 4), np.linspace(-1.0, 1.0, 4))
    true_grids = [x_grid + y_grid, x_grid - y_grid]
    pred_grids = [grid + 0.1 for grid in true_grids]

    paths, rel_err = plot_2d_time_slice_files(
        "demo",
        x_grid,
        y_grid,
        true_grids,
        pred_grids,
        [0.5, 1.0],
        tmp_path,
    )

    assert rel_err > 0.0
    assert [path.name for path in paths] == [
        "demo_t0p500_true_smoke.png",
        "demo_t0p500_sol_smoke.png",
        "demo_t0p500_abs_error_smoke.png",
        "demo_t1p000_true_smoke.png",
        "demo_t1p000_sol_smoke.png",
        "demo_t1p000_abs_error_smoke.png",
    ]
    assert all(path.exists() for path in paths)


def test_lshape_default_slices_follow_reference_final_time(tmp_path):
    ref_path = tmp_path / "lshape_reference.npz"
    times = np.linspace(0.0, 1.0, 11)
    x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 2), np.linspace(-1.0, 1.0, 2))
    snapshots = np.arange(len(times) * 4, dtype=float).reshape(len(times), 2, 2)
    np.savez(
        ref_path,
        alpha=1.8,
        diffusion_scale=0.25,
        t_final=1.0,
        times=times,
        x_grid=x_grid,
        y_grid=y_grid,
        snapshots=snapshots,
    )

    args = type("Args", (), {"lshape_data": ref_path, "time_slices": None})()
    _, _, selected, meta = lshape_reference_slices(args)

    assert meta["alpha"] == 1.8
    assert meta["t_final"] == 1.0
    assert [time_value for time_value, _ in selected] == [0.5, 1.0]


def test_alpha_specific_reference_paths_use_tokens(tmp_path):
    lshape_dir = tmp_path / "lshape"
    lshape_dir.mkdir()
    lshape_ref = lshape_dir / "lshape_reference_alpha1p25.npz"
    lshape_ref.touch()

    burgers_dir = tmp_path / "burgers"
    burgers_dir.mkdir()
    burgers_ref = burgers_dir / "burgers_175.npz"
    burgers_ref.touch()

    assert float_token(1.5) == "1p50"
    assert resolve_lshape_data_path(None, 1.25, lshape_dir) == lshape_ref
    assert resolve_burgers_data_path(None, 1.75, burgers_dir) == burgers_ref
