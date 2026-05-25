import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "peaks" / "3.5"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interval" / "3.5"


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def compute_window_nn_intervals(peak_indices, bvp_hz):
    peak_indices = np.asarray(peak_indices, dtype=np.int64)
    if len(peak_indices) < 2:
        return np.empty((0,), dtype=np.float64)
    return np.diff(peak_indices) / bvp_hz


def process_subject(npz_path, output_path, overwrite=False):
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path} "
            "(use --overwrite to replace it)"
        )

    data = np.load(npz_path, allow_pickle=True)
    peak_indices = data["peak_indices"]
    bvp_hz = int(np.asarray(data["bvp_hz"]).item())

    nn_intervals_sec = []
    nn_counts = []
    for window_peak_indices in peak_indices:
        intervals = compute_window_nn_intervals(window_peak_indices, bvp_hz)
        nn_intervals_sec.append(intervals)
        nn_counts.append(len(intervals))

    nn_intervals_sec = np.asarray(nn_intervals_sec, dtype=object)
    nn_counts = np.asarray(nn_counts, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        nn_intervals_sec=nn_intervals_sec,
        nn_intervals_ms=nn_intervals_sec * 1000.0,
        nn_counts=nn_counts,
        y=data["y"],
        starts=data["starts"],
        ends=data["ends"],
        start_seconds=data["start_seconds"],
        end_seconds=data["end_seconds"],
        subject=data["subject"],
        bvp_hz=data["bvp_hz"],
        window_seconds=data["window_seconds"],
        stride_seconds=data["stride_seconds"],
        window_function=data["window_function"] if "window_function" in data else np.asarray("unknown"),
        source_npz=np.asarray(str(npz_path)),
    )

    all_intervals = (
        np.concatenate([np.asarray(values, dtype=np.float64) for values in nn_intervals_sec])
        if len(nn_intervals_sec) and np.sum(nn_counts) > 0
        else np.empty((0,), dtype=np.float64)
    )

    return {
        "subject": str(np.asarray(data["subject"]).item()),
        "input_npz": str(npz_path),
        "output_npz": str(output_path),
        "windows": int(len(peak_indices)),
        "total_intervals": int(len(all_intervals)),
        "min_intervals_per_window": int(np.min(nn_counts)) if len(nn_counts) else 0,
        "max_intervals_per_window": int(np.max(nn_counts)) if len(nn_counts) else 0,
        "mean_intervals_per_window": float(np.mean(nn_counts)) if len(nn_counts) else 0.0,
        "mean_nn_sec": float(np.mean(all_intervals)) if len(all_intervals) else 0.0,
        "std_nn_sec": float(np.std(all_intervals)) if len(all_intervals) else 0.0,
    }


def write_summary(summary_path, rows):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "input_npz",
        "output_npz",
        "windows",
        "total_intervals",
        "min_intervals_per_window",
        "max_intervals_per_window",
        "mean_intervals_per_window",
        "mean_nn_sec",
        "std_nn_sec",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute NN intervals from detected BVP peak indices."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing detected peak NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write NN interval NPZ files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing interval NPZ files and summary.csv.",
    )
    args = parser.parse_args()

    npz_paths = sorted(args.input_dir.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no peak NPZ files found in {args.input_dir}")

    if args.output_dir.exists() and not args.overwrite:
        existing_outputs = list(args.output_dir.glob("*.npz")) + list(
            args.output_dir.glob("summary.csv")
        )
        if existing_outputs:
            raise FileExistsError(
                f"output directory already contains files: {args.output_dir} "
                "(use --overwrite to replace them)"
            )

    rows = []
    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print("method: NN interval = diff(peak_indices) / bvp_hz")
    print(f"subjects: {len(npz_paths)}")
    print()

    for npz_path in npz_paths:
        output_path = args.output_dir / npz_path.name
        row = process_subject(npz_path, output_path, overwrite=args.overwrite)
        rows.append(row)

        print(f"[ok] {row['subject']}")
        print(f"  output: {row['output_npz']}")
        print(f"  windows: {row['windows']}")
        print(f"  total intervals: {row['total_intervals']}")
        print(f"  mean NN: {row['mean_nn_sec']:.4f}s")
        print()

    summary_path = args.output_dir / "summary.csv"
    write_summary(summary_path, rows)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
