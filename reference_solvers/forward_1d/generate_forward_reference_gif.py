import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pymittagleffler import mittag_leffler


def float_token(value):
    return f"{value:.2f}".replace(".", "p")


def forward_solution(alpha, lam, k, a, b, t, x):
    t_col = t.reshape(-1, 1)
    x_row = x.reshape(1, -1)
    z = -lam * np.power(t_col, alpha)
    time_part = (
        a * np.real(mittag_leffler(z, alpha, 1.0))
        + b * t_col * np.real(mittag_leffler(z, alpha, 2.0))
    )
    return np.sin(k * np.pi * x_row) * time_part


def render_line_frame(x, u, title, ylim):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120, layout="constrained")
    ax.plot(x, u, color="#a51c30", lw=2.2)
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


def write_summary(outdir, row):
    path = outdir / "forward_reference_summary.csv"
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
                "t_min",
                "t_max",
                "lambda",
                "k",
                "a",
                "b",
            ],
        )
        writer.writeheader()
        writer.writerow(row)
    return path


def make_forward_gif(args):
    t = np.linspace(args.t_min, args.t_max, args.frames)
    x = np.linspace(args.x_min, args.x_max, args.x_points)
    u = forward_solution(args.alpha, args.lam, args.k, args.a, args.b, t, x)
    margin = 0.06 * (float(np.max(u)) - float(np.min(u)) + 1e-12)
    ylim = (float(np.min(u)) - margin, float(np.max(u)) + margin)

    frames = []
    for idx, t_value in enumerate(t):
        title = (
            rf"Forward analytic reference, $\alpha={args.alpha:.2f}$, "
            rf"$k={args.k}$, $\lambda={args.lam:g}$, $t={t_value:.3f}$"
        )
        frames.append(render_line_frame(x, u[idx], title, ylim))

    gif_path = args.outdir / f"forward_alpha{float_token(args.alpha)}_reference.gif"
    save_gif(frames, gif_path, args.duration)
    row = {
        "case": "forward",
        "alpha": f"{args.alpha:.2f}",
        "source": "analytic Mittag-Leffler solution",
        "gif": str(gif_path),
        "frames": len(frames),
        "x_points": len(x),
        "t_min": float(t[0]),
        "t_max": float(t[-1]),
        "lambda": args.lam,
        "k": args.k,
        "a": args.a,
        "b": args.b,
    }
    summary = write_summary(args.outdir, row)
    print(f"saved: {gif_path}")
    print(f"summary: {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("data/reference_1d"))
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--alpha", type=float, default=1.75)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=-0.5)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=2.0)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--x-points", type=int, default=401)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    make_forward_gif(args)


if __name__ == "__main__":
    main()
