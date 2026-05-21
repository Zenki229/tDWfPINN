import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.generate_smoke_results import (  # noqa: E402
    make_burgers,
    make_forward,
    make_irregular_hole,
    make_lshape,
)


MAKERS = {
    "forward": make_forward,
    "burgers": make_burgers,
    "irregular_hole": make_irregular_hole,
    "lshape": make_lshape,
}


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def method_parts(method):
    quad, type_token = method.split("-")
    return quad, type_token


def file_token(value):
    return value.lower().replace("-", "_")


def default_outdir():
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs") / "stress_results" / stamp


def run_one(case, method, args):
    quad_family, type_token = method_parts(method)
    quad = args.gj_quad if quad_family == "GJ" else args.mc_quad
    batch_in = args.batch_in_2d if case in {"irregular_hole", "lshape"} else args.batch_in_1d

    smoke_args = SimpleNamespace(
        case=case,
        steps=args.steps,
        quad=quad,
        method=method,
        output_prefix=f"{file_token(case)}_{file_token(method)}",
        batch_in=batch_in,
        batch_bd=args.batch_bd,
        batch_init=args.batch_init,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        grid_1d=args.grid_1d,
        grid_2d=args.grid_2d,
        time_slices=args.time_slices,
        seed=args.seed,
        outdir=args.outdir,
        burgers_data=args.burgers_data,
        lshape_data=args.lshape_data,
    )
    row = MAKERS[case](smoke_args)
    elapsed = float(row["elapsed_seconds"])
    row.update({
        "quadrature": quad_family,
        "type": type_token,
        "steps": args.steps,
        "quad_points": quad,
        "batch_in": batch_in,
        "batch_bd": args.batch_bd,
        "batch_init": args.batch_init,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "average_step_seconds": elapsed / max(args.steps, 1),
    })
    return row


def write_summary(rows, outdir):
    path = outdir / "summary.csv"
    fieldnames = [
        "case",
        "method",
        "quadrature",
        "type",
        "steps",
        "quad_points",
        "batch_in",
        "batch_bd",
        "batch_init",
        "hidden_dim",
        "num_layers",
        "elapsed_seconds",
        "average_step_seconds",
        "loss",
        "relative_error",
        "path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    f"{row[key]:.8e}"
                    if key in {"elapsed_seconds", "average_step_seconds", "loss", "relative_error"}
                    else row[key]
                )
                for key in fieldnames
            })
    return path


def write_timing_pivot(rows, outdir, cases, methods):
    path = outdir / "timing_seconds_pivot.csv"
    by_key = {(row["case"], row["method"]): float(row["elapsed_seconds"]) for row in rows}
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", *methods])
        for case in cases:
            writer.writerow([
                case,
                *[
                    f"{by_key[(case, method)]:.8f}"
                    if (case, method) in by_key
                    else ""
                    for method in methods
                ],
            ])
    return path


def plot_timing(rows, outdir, cases, methods):
    by_key = {(row["case"], row["method"]): float(row["elapsed_seconds"]) for row in rows}
    x = np.arange(len(cases))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))

    fig, ax = plt.subplots(figsize=(10.5, 5.0), layout="constrained")
    for offset, method in zip(offsets, methods):
        values = [by_key.get((case, method), np.nan) for case in cases]
        ax.bar(x + offset, values, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15, ha="right")
    ax.set_ylabel("seconds / 5000 steps")
    ax.set_title("Stress-test wall time by PDE and quadrature method")
    ax.legend(ncols=4)
    ax.grid(axis="y", alpha=0.25)
    path = outdir / "timing_seconds.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def choose_abs_error_path(row):
    paths = [Path(piece) for piece in str(row["path"]).split(";")]
    abs_paths = [path for path in paths if "abs_error" in path.name]
    if not abs_paths:
        return paths[-1]
    final_time = [path for path in abs_paths if "t1p000" in path.name or "t2p000" in path.name]
    return final_time[-1] if final_time else abs_paths[-1]


def make_preview_sheet(rows, outdir, cases, methods):
    lookup = {(row["case"], row["method"]): row for row in rows}
    thumb_w, thumb_h = 260, 195
    label_h = 44
    margin = 16
    sheet_w = margin + len(methods) * (thumb_w + margin)
    sheet_h = margin + len(cases) * (thumb_h + label_h + margin)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row_idx, case in enumerate(cases):
        for col_idx, method in enumerate(methods):
            row = lookup.get((case, method))
            if row is None:
                continue
            path = choose_abs_error_path(row)
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            x0 = margin + col_idx * (thumb_w + margin)
            y0 = margin + row_idx * (thumb_h + label_h + margin)
            draw.text((x0, y0), f"{case} / {method}", fill="black", font=font)
            draw.text(
                (x0, y0 + 14),
                f"{float(row['elapsed_seconds']):.1f}s, rel {float(row['relative_error']):.2e}",
                fill="black",
                font=font,
            )
            sheet.paste(img, (x0, y0 + label_h))

    path = outdir / "abs_error_preview.png"
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="forward,burgers,irregular_hole,lshape",
        help="comma-separated PDE cases",
    )
    parser.add_argument(
        "--methods",
        default="GJ-I,GJ-II,MC-I,MC-II",
        help="comma-separated quadrature methods",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--gj-quad", type=int, default=3)
    parser.add_argument("--mc-quad", type=int, default=3)
    parser.add_argument("--batch-in-1d", type=int, default=8)
    parser.add_argument("--batch-in-2d", type=int, default=4)
    parser.add_argument("--batch-bd", type=int, default=2)
    parser.add_argument("--batch-init", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--grid-1d", type=int, default=80)
    parser.add_argument("--grid-2d", type=int, default=90)
    parser.add_argument("--time-slices", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--burgers-data", type=Path, default=Path("data/burgers_150.npz"))
    parser.add_argument(
        "--lshape-data",
        type=Path,
        default=Path("data/lshape/lshape_reference.npz"),
    )
    args = parser.parse_args()
    args.outdir = args.outdir or default_outdir()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = parse_csv_list(args.cases)
    methods = parse_csv_list(args.methods)
    rows = []
    for case in cases:
        if case not in MAKERS:
            raise ValueError(f"Unknown case: {case}")
        for method in methods:
            if method not in {"GJ-I", "GJ-II", "MC-I", "MC-II"}:
                raise ValueError(f"Unknown method: {method}")
            print(f"[*] running case={case}, method={method}, steps={args.steps}")
            row = run_one(case, method, args)
            rows.append(row)
            print(
                "[*] done "
                f"case={case}, method={method}, "
                f"elapsed={float(row['elapsed_seconds']):.3f}s, "
                f"loss={float(row['loss']):.3e}, "
                f"rel={float(row['relative_error']):.3e}"
            )

    summary = write_summary(rows, args.outdir)
    pivot = write_timing_pivot(rows, args.outdir, cases, methods)
    timing_plot = plot_timing(rows, args.outdir, cases, methods)
    preview = make_preview_sheet(rows, args.outdir, cases, methods)

    print(f"summary={summary}")
    print(f"timing_pivot={pivot}")
    print(f"timing_plot={timing_plot}")
    print(f"preview={preview}")


if __name__ == "__main__":
    main()
