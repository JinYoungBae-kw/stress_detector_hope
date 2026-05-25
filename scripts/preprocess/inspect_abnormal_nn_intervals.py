import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "interval" / "4.0"
OUTPUT_CSV = PROJECT_ROOT / "data" / "interval" / "4.0.csv"

LOWER_SEC = 0.30
UPPER_SEC = 2.00


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def scalar_string(value):
    arr = np.asarray(value)
    return str(arr.item() if arr.shape == () else arr)


def analyze_window(nn_sec, lower_sec, upper_sec):
    nn_sec = np.asarray(nn_sec, dtype=np.float64).reshape(-1)
    short_mask = nn_sec < lower_sec
    long_mask = nn_sec > upper_sec
    abnormal_mask = short_mask | long_mask

    total = int(len(nn_sec))
    short_count = int(np.sum(short_mask))
    long_count = int(np.sum(long_mask))
    abnormal_count = int(np.sum(abnormal_mask))

    return {
        "nn_count": total,
        "short_count": short_count,
        "long_count": long_count,
        "abnormal_count": abnormal_count,
        "abnormal_ratio": float(abnormal_count / total) if total else 0.0,
        "min_nn_sec": float(np.min(nn_sec)) if total else np.nan,
        "max_nn_sec": float(np.max(nn_sec)) if total else np.nan,
        "mean_nn_sec": float(np.mean(nn_sec)) if total else np.nan,
    }


def process_subject(npz_path, lower_sec, upper_sec):
    data = np.load(npz_path, allow_pickle=True)
    nn_intervals_sec = data["nn_intervals_sec"]
    y = data["y"]
    subject = scalar_string(data["subject"]) if "subject" in data else npz_path.stem

    rows = []
    for window_index, nn_sec in enumerate(nn_intervals_sec):
        stats = analyze_window(nn_sec, lower_sec, upper_sec)
        row = {
            "subject": subject,
            "window_index": window_index,
            "label": int(y[window_index]),
            "start_seconds": float(data["start_seconds"][window_index]),
            "end_seconds": float(data["end_seconds"][window_index]),
            **stats,
        }
        rows.append(row)

    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "window_index",
        "label",
        "start_seconds",
        "end_seconds",
        "nn_count",
        "short_count",
        "long_count",
        "abnormal_count",
        "abnormal_ratio",
        "min_nn_sec",
        "max_nn_sec",
        "mean_nn_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_subject_summary(subject, rows):
    total_nn = sum(row["nn_count"] for row in rows)
    total_abnormal = sum(row["abnormal_count"] for row in rows)
    total_short = sum(row["short_count"] for row in rows)
    total_long = sum(row["long_count"] for row in rows)
    ratio = total_abnormal / total_nn if total_nn else 0.0

    print(f"[subject] {subject}")
    print(
        "  total: "
        f"nn={total_nn}, "
        f"abnormal={total_abnormal} ({ratio * 100:.2f}%), "
        f"short={total_short}, "
        f"long={total_long}"
    )

    for row in rows:
        print(
            "  window "
            f"{row['window_index']:>3} | "
            f"label={row['label']} | "
            f"{row['start_seconds']:.1f}-{row['end_seconds']:.1f}s | "
            f"nn={row['nn_count']} | "
            f"abnormal={row['abnormal_count']} "
            f"({row['abnormal_ratio'] * 100:.2f}%) | "
            f"short={row['short_count']} | "
            f"long={row['long_count']} | "
            f"min={row['min_nn_sec']:.3f}s | "
            f"max={row['max_nn_sec']:.3f}s | "
            f"mean={row['mean_nn_sec']:.3f}s"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect abnormal NN intervals outside a physiological range."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing subject NN interval NPZ files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="CSV path to write window-level abnormal NN interval summary.",
    )
    parser.add_argument(
        "--lower-sec",
        type=float,
        default=LOWER_SEC,
        help=f"Lower physiological NN interval bound in seconds. Default: {LOWER_SEC}.",
    )
    parser.add_argument(
        "--upper-sec",
        type=float,
        default=UPPER_SEC,
        help=f"Upper physiological NN interval bound in seconds. Default: {UPPER_SEC}.",
    )
    args = parser.parse_args()

    if args.lower_sec <= 0 or args.upper_sec <= args.lower_sec:
        raise ValueError("Require 0 < lower-sec < upper-sec")

    npz_paths = sorted(args.input_dir.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no interval NPZ files found in {args.input_dir}")

    all_rows = []
    print(f"input_dir: {args.input_dir}")
    print(f"output_csv: {args.output_csv}")
    print(
        "physiological range: "
        f"{args.lower_sec:.3f}-{args.upper_sec:.3f}s "
        f"({60 / args.upper_sec:.1f}-{60 / args.lower_sec:.1f} BPM)"
    )
    print(f"subjects: {len(npz_paths)}")
    print()

    for npz_path in npz_paths:
        rows = process_subject(npz_path, args.lower_sec, args.upper_sec)
        all_rows.extend(rows)
        subject = rows[0]["subject"] if rows else npz_path.stem
        print_subject_summary(subject, rows)

    write_csv(args.output_csv, all_rows)

    total_nn = sum(row["nn_count"] for row in all_rows)
    total_abnormal = sum(row["abnormal_count"] for row in all_rows)
    total_short = sum(row["short_count"] for row in all_rows)
    total_long = sum(row["long_count"] for row in all_rows)
    ratio = total_abnormal / total_nn if total_nn else 0.0

    print("[overall]")
    print(
        f"  nn={total_nn}, "
        f"abnormal={total_abnormal} ({ratio * 100:.2f}%), "
        f"short={total_short}, "
        f"long={total_long}"
    )
    print(f"summary: {args.output_csv}")


if __name__ == "__main__":
    main()
