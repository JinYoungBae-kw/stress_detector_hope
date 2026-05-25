from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUBJECT = "S2"
WINDOW_INDEX = 0

WINDOW_NPZ_PATH = rf"C:\ppg_stress\data\windowed\new\{SUBJECT}.npz"
PEAK_NPZ_PATH = rf"C:\ppg_stress\data\peaks\new_threshold\{SUBJECT}.npz"

BVP_HZ = 64

LABEL_NAMES = {
    0: "stress",
    1: "non-stress",
}


def scalar_string(value):
    arr = np.asarray(value)
    return str(arr.item() if arr.shape == () else arr)


def main():
    window_data = np.load(Path(WINDOW_NPZ_PATH), allow_pickle=True)
    peak_data = np.load(Path(PEAK_NPZ_PATH), allow_pickle=True)

    X = window_data["X"]
    y = window_data["y"]

    if len(X) == 0:
        raise ValueError(f"no windows found in {WINDOW_NPZ_PATH}")
    if WINDOW_INDEX < 0 or WINDOW_INDEX >= len(X):
        raise IndexError(f"WINDOW_INDEX must be between 0 and {len(X) - 1}")

    subject = scalar_string(window_data["subject"])
    bvp_hz = int(window_data["bvp_hz"]) if "bvp_hz" in window_data else BVP_HZ
    window = X[WINDOW_INDEX]
    label = int(y[WINDOW_INDEX])
    label_name = LABEL_NAMES.get(label, "unknown")
    start_seconds = float(window_data["start_seconds"][WINDOW_INDEX])
    end_seconds = float(window_data["end_seconds"][WINDOW_INDEX])

    peak_indices = peak_data["peak_indices"][WINDOW_INDEX].astype(np.int64)
    peak_values = peak_data["peak_values"][WINDOW_INDEX].astype(np.float64)
    threshold = float(peak_data["thresholds"][WINDOW_INDEX])

    time = np.arange(len(window)) / bvp_hz
    peak_times = peak_indices / bvp_hz

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, window, color="#1f77b4", linewidth=0.8, label="BVP window")
    ax.scatter(
        peak_times,
        peak_values,
        color="#d62728",
        s=18,
        label=f"peaks ({len(peak_indices)})",
        zorder=3,
    )
    ax.axhline(threshold, color="#666666", linestyle="--", linewidth=1.0, label="window mean")

    ax.set_title(
        f"{subject} window {WINDOW_INDEX} | {label}: {label_name} | "
        f"{start_seconds:.1f}s - {end_seconds:.1f}s"
    )
    ax.set_xlabel("Window time (seconds)")
    ax.set_ylabel("BVP value")
    ax.set_xlim(0, len(window) / bvp_hz)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
