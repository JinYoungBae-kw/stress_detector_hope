from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUBJECT = "S2"
WINDOW_INDEX = 0
UNIT = "sec"  # "sec" or "ms"

INTERVAL_NPZ_PATH = rf"C:\ppg_stress\data\interval\bvp_360s_stride30s_hanning\{SUBJECT}.npz"

LABEL_NAMES = {
    0: "stress",
    1: "non-stress",
}


def scalar_string(value):
    arr = np.asarray(value)
    return str(arr.item() if arr.shape == () else arr)


def main():
    data = np.load(Path(INTERVAL_NPZ_PATH), allow_pickle=True)

    key = "nn_intervals_ms" if UNIT == "ms" else "nn_intervals_sec"
    unit_label = "ms" if UNIT == "ms" else "sec"

    intervals = data[key][WINDOW_INDEX].astype(np.float64)
    y = data["y"]

    if WINDOW_INDEX < 0 or WINDOW_INDEX >= len(y):
        raise IndexError(f"WINDOW_INDEX must be between 0 and {len(y) - 1}")

    subject = scalar_string(data["subject"])
    label = int(y[WINDOW_INDEX])
    label_name = LABEL_NAMES.get(label, "unknown")
    start_seconds = float(data["start_seconds"][WINDOW_INDEX])
    end_seconds = float(data["end_seconds"][WINDOW_INDEX])

    if len(intervals) == 0:
        raise ValueError(f"window {WINDOW_INDEX} has no NN intervals")

    beat_order = np.arange(1, len(intervals) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"{subject} window {WINDOW_INDEX} | {label}: {label_name} | "
        f"{start_seconds:.1f}s - {end_seconds:.1f}s"
    )

    axes[0].plot(beat_order, intervals, color="#1f77b4", linewidth=0.9, marker=".", markersize=3)
    axes[0].set_title("NN interval sequence")
    axes[0].set_xlabel("Beat order")
    axes[0].set_ylabel(f"NN interval ({unit_label})")
    axes[0].grid(alpha=0.25)

    axes[1].hist(intervals, bins=40, color="#2ca02c", alpha=0.85, edgecolor="white")
    axes[1].set_title("NN interval histogram")
    axes[1].set_xlabel(f"NN interval ({unit_label})")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
