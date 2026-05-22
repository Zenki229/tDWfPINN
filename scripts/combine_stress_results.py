import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.run_stress_tests import (
    make_preview_sheet,
    parse_alpha_list,
    float_token,
    plot_timing,
    write_summary,
    write_timing_pivot,
)


FLOAT_KEYS = {"alpha", "elapsed_seconds", "average_step_seconds", "loss", "relative_error"}
INT_KEYS = {
    "steps",
    "quad_points",
    "batch_in",
    "batch_bd",
    "batch_init",
    "hidden_dim",
    "num_layers",
}


def read_rows(summary_files):
    rows = []
    for path in summary_files:
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                row.setdefault("alpha", "nan")
                for key in FLOAT_KEYS:
                    row[key] = float(row[key])
                for key in INT_KEYS:
                    row[key] = int(row[key])
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument(
        "--cases",
        default="forward,burgers,irregular_hole,lshape",
    )
    parser.add_argument("--methods", default="GJ-I,GJ-II,MC-I,MC-II")
    parser.add_argument("--alphas", default="1.25,1.5,1.75")
    args = parser.parse_args()

    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    alphas = parse_alpha_list(args.alphas)
    rows = read_rows(args.summary)
    alpha_order = {float_token(alpha): idx for idx, alpha in enumerate(alphas)}
    rows.sort(key=lambda row: (
        cases.index(row["case"]),
        alpha_order.get(float_token(row["alpha"]), len(alpha_order)),
        methods.index(row["method"]),
    ))

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = write_summary(rows, args.outdir)
    pivot = write_timing_pivot(rows, args.outdir, cases, methods, alphas)
    timing_plot = plot_timing(rows, args.outdir, cases, methods, alphas)
    preview = make_preview_sheet(rows, args.outdir, cases, methods, alphas)

    print(f"summary={summary}")
    print(f"timing_pivot={pivot}")
    print(f"timing_plot={timing_plot}")
    print(f"preview={preview}")


if __name__ == "__main__":
    main()
