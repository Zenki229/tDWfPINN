import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pymittagleffler import mittag_leffler


def float_token(value):
    return f"{value:.2f}".replace(".", "p")


def alpha_from_burgers_path(path):
    match = re.search(r"burgers_(\d+)", path.stem)
    if not match:
        return np.nan
    return int(match.group(1)) / 100.0


def select_frame_indices(n_times, n_frames):
    return np.unique(np.linspace(0, n_times - 1, min(n_frames, n_times), dtype=int))


def render_line_frame(x, u, title, ylim, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120, layout="constrained")
    ax.plot(x, u, color=color, lw=2.2)
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


def make_burgers_gif(path, outdir, n_frames, duration_ms):
    data = np.load(path)
    t = np.asarray(data["t"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    u = np.asarray(data["u"], dtype=float)
    if u.shape == (len(x), len(t)):
        u = u.T
    if u.shape != (len(t), len(x)):
        raise ValueError(f"Unexpected Burgers data shape in {path}: {u.shape}")

    alpha = alpha_from_burgers_path(path)
    indices = select_frame_indices(len(t), n_frames)
    margin = 0.06 * (float(np.max(u)) - float(np.min(u)) + 1e-12)
    ylim = (float(np.min(u)) - margin, float(np.max(u)) + margin)
    frames = []
    for idx in indices:
        title = rf"Burgers reference, $\alpha={alpha:.2f}$, $t={t[idx]:.3f}$"
        frames.append(render_line_frame(x, u[idx], title, ylim, color="#005f9e"))

    gif_path = outdir / f"burgers_alpha{float_token(alpha)}_reference.gif"
    save_gif(frames, gif_path, duration_ms)
    return {
        "case": "burgers",
        "alpha": f"{alpha:.2f}",
        "source": str(path),
        "gif": str(gif_path),
        "frames": len(frames),
        "x_points": len(x),
        "t_min": float(t[indices[0]]),
        "t_max": float(t[indices[-1]]),
    }


def forward_solution(alpha, lam, k, a, b, t, x):
    t_col = t.reshape(-1, 1)
    x_row = x.reshape(1, -1)
    z = -lam * np.power(t_col, alpha)
    time_part = (
        a * np.real(mittag_leffler(z, alpha, 1.0))
        + b * t_col * np.real(mittag_leffler(z, alpha, 2.0))
    )
    return np.sin(k * np.pi * x_row) * time_part


def make_forward_gif(args):
    t = np.linspace(args.forward_t_min, args.forward_t_max, args.frames)
    x = np.linspace(args.forward_x_min, args.forward_x_max, args.forward_x_points)
    u = forward_solution(
        args.forward_alpha,
        args.forward_lambda,
        args.forward_k,
        args.forward_a,
        args.forward_b,
        t,
        x,
    )
    margin = 0.06 * (float(np.max(u)) - float(np.min(u)) + 1e-12)
    ylim = (float(np.min(u)) - margin, float(np.max(u)) + margin)
    frames = []
    for idx, t_value in enumerate(t):
        title = (
            rf"Forward reference, $\alpha={args.forward_alpha:.2f}$, "
            rf"$k={args.forward_k}$, $\lambda={args.forward_lambda:g}$, "
            rf"$t={t_value:.3f}$"
        )
        frames.append(render_line_frame(x, u[idx], title, ylim, color="#a51c30"))

    gif_path = args.outdir / f"forward_alpha{float_token(args.forward_alpha)}_reference.gif"
    save_gif(frames, gif_path, args.duration)
    return {
        "case": "forward",
        "alpha": f"{args.forward_alpha:.2f}",
        "source": "analytic Mittag-Leffler solution",
        "gif": str(gif_path),
        "frames": len(frames),
        "x_points": len(x),
        "t_min": float(t[0]),
        "t_max": float(t[-1]),
    }


def write_summary(outdir, rows):
    path = outdir / "reference_1d_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "alpha", "source", "gif", "frames", "x_points", "t_min", "t_max"],
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
    parser.add_argument("--skip-burgers", action="store_true")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--forward-alpha", type=float, default=1.75)
    parser.add_argument("--forward-lambda", type=float, default=1.0)
    parser.add_argument("--forward-k", type=int, default=1)
    parser.add_argument("--forward-a", type=float, default=1.0)
    parser.add_argument("--forward-b", type=float, default=-0.5)
    parser.add_argument("--forward-t-min", type=float, default=0.0)
    parser.add_argument("--forward-t-max", type=float, default=2.0)
    parser.add_argument("--forward-x-min", type=float, default=0.0)
    parser.add_argument("--forward-x-max", type=float, default=1.0)
    parser.add_argument("--forward-x-points", type=int, default=401)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    if not args.skip_burgers:
        burgers_paths = args.burgers_data or sorted(Path("data").glob("burgers_*.npz"))
        for path in burgers_paths:
            rows.append(make_burgers_gif(path, args.outdir, args.frames, args.duration))

    if not args.skip_forward:
        rows.append(make_forward_gif(args))

    summary = write_summary(args.outdir, rows)
    for row in rows:
        print(f"saved: {row['gif']}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
