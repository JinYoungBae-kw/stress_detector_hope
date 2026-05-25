from pathlib import Path

import numpy as np


FEATURE_DIR = r"C:\ppg_stress\data\features\bvp_360s_stride30s_hanning"

LABEL_NAMES = {
    0: "stress",
    1: "non-stress",
}


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def print_feature_stats(feature_names, X):
    for index, name in enumerate(feature_names):
        values = X[:, index]
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            print(f"    {name}: all NaN/inf")
            continue
        print(
            f"    {name}: "
            f"mean={np.mean(finite_values):.6g}, "
            f"std={np.std(finite_values):.6g}, "
            f"min={np.min(finite_values):.6g}, "
            f"max={np.max(finite_values):.6g}"
        )


def main():
    feature_dir = Path(FEATURE_DIR)
    npz_paths = sorted(feature_dir.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no feature NPZ files found in {feature_dir}")

    total_windows = 0
    total_nan_rows = 0
    total_label_counts = {}

    print(f"feature_dir: {feature_dir}")
    print(f"subjects: {len(npz_paths)}")
    print()

    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        feature_names = [str(name) for name in data["feature_names"]]
        nan_rows = data["nan_rows"] if "nan_rows" in data else np.any(~np.isfinite(X), axis=1)

        unique_labels, counts = np.unique(y, return_counts=True)
        label_counts = dict(zip(unique_labels.astype(int), counts.astype(int)))
        total_windows += len(y)
        total_nan_rows += int(np.sum(nan_rows))
        for label, count in label_counts.items():
            total_label_counts[label] = total_label_counts.get(label, 0) + count

        print(f"[{npz_path.stem}]")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  NaN rows: {int(np.sum(nan_rows))}")
        print("  label distribution:")
        for label, count in sorted(label_counts.items()):
            label_name = LABEL_NAMES.get(label, "unknown")
            print(f"    {label}: {count} ({label_name})")
        print("  feature stats:")
        print_feature_stats(feature_names, X)
        print()

    print("[total]")
    print(f"  windows: {total_windows}")
    print(f"  NaN rows: {total_nan_rows}")
    print("  label distribution:")
    for label, count in sorted(total_label_counts.items()):
        label_name = LABEL_NAMES.get(label, "unknown")
        print(f"    {label}: {count} ({label_name})")


if __name__ == "__main__":
    main()
