import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def float_token(value):
    return f"{value:.2f}".replace(".", "p")


def alpha_from_burgers_path(path):
    match = re.search(r"burgers_(\d+)", path.stem)
    if not match:
        return np.nan
    return int(match.group(1)) / 100.0


def select_frame_indices(n_times, n_frames):
    return np.unique(np.linspace(0, n_times - 1, min(n_frames, n_times), dtype=int))


def render_line_frame(x, u, title, ylim):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120, layout="constrained")
    ax.plot(x, u, color="#005f9e", lw=2.2)
    ax.axhline(0.0, color="0.25", lw=0.8, alpha=0.55)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(title)
    ax.grid(True, color="0.88", lw=0.6)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    image = Image.fromarray(frame[:, :, :3])
    plt.close(fig)
    return image


def save_gif(frames, path, duration_ms):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def relative_l2_error(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-14))


def orient_burgers_solution(u, x, t):
    """Return u as (time, x), using the initial condition to resolve square data.

    The stored reference arrays are interpreted as data["u"] with shape
    (t, x). A line-frame GIF at one time level must use u[idx_t, :].
    """
    expected_u0 = -np.sin(np.pi * x)
    candidates = []
    if u.shape == (len(t), len(x)):
        candidates.append(("tx", u, relative_l2_error(u[0, :], expected_u0)))
    if u.shape == (len(x), len(t)):
        candidates.append(("xt", u.T, relative_l2_error(u[:, 0], expected_u0)))
    if not candidates:
        raise ValueError(
            f"Unexpected Burgers data shape {u.shape}; expected "
            f"({len(t)}, {len(x)}) or ({len(x)}, {len(t)})"
        )
    orientation, oriented, init_error = min(candidates, key=lambda item: item[2])
    return orientation, oriented, init_error


def make_burgers_gif(path, outdir, n_frames, duration_ms):
    data = np.load(path)
    t = np.asarray(data["t"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    u_raw = np.asarray(data["u"], dtype=float)
    orientation, u, init_error = orient_burgers_solution(u_raw, x, t)

    alpha = alpha_from_burgers_path(path)
    indices = select_frame_indices(len(t), n_frames)
    margin = 0.06 * (float(np.max(u)) - float(np.min(u)) + 1e-12)
    ylim = (float(np.min(u)) - margin, float(np.max(u)) + margin)
    frames = []
    for idx in indices:
        title = rf"Burgers reference, $\alpha={alpha:.2f}$, $t={t[idx]:.3f}$"
        frames.append(render_line_frame(x, u[idx], title, ylim))

    gif_path = outdir / f"burgers_alpha{float_token(alpha)}_reference.gif"
    save_gif(frames, gif_path, duration_ms)
    return {
        "case": "burgers",
        "alpha": f"{alpha:.2f}",
        "source": str(path),
        "gif": str(gif_path),
        "frames": len(frames),
        "x_points": len(x),
        "t_points": len(t),
        "t_min": float(t[indices[0]]),
        "t_max": float(t[indices[-1]]),
        "orientation": orientation,
        "initial_condition_relative_l2": f"{init_error:.6e}",
    }


def write_summary(outdir, rows):
    path = outdir / "burgers_reference_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "alpha",
                "source",
                "gif",
                "frames",
                "x_points",
                "t_points",
                "t_min",
                "t_max",
                "orientation",
                "initial_condition_relative_l2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("data/reference_1d"))
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--burgers-data", type=Path, nargs="*", default=None)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    burgers_paths = args.burgers_data or sorted(Path("data").glob("burgers_*.npz"))
    rows = [
        make_burgers_gif(path, args.outdir, args.frames, args.duration)
        for path in burgers_paths
    ]
    summary = write_summary(args.outdir, rows)
    for row in rows:
        print(
            f"saved: {row['gif']} "
            f"(orientation={row['orientation']}, "
            f"init_rel_l2={row['initial_condition_relative_l2']})"
        )
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
