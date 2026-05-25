from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUBJECT = "S2"
WINDOW_INDEX = 0

RECT_NPZ_PATH = rf"C:\ppg_stress\data\windowed\bvp_360s_stride30s_rect\{SUBJECT}.npz"
HANNING_NPZ_PATH = rf"C:\ppg_stress\data\windowed\bvp_360s_stride30s\{SUBJECT}.npz"

BVP_HZ = 64

LABEL_NAMES = {
    0: "stress",
    1: "non-stress",
}


def scalar_string(value):
    arr = np.asarray(value)
    return str(arr.item() if arr.shape == () else arr)


def load_window(npz_path, window_index):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    if len(X) == 0:
        raise ValueError(f"no windows found in {npz_path}")
    if window_index < 0 or window_index >= len(X):
        raise IndexError(f"WINDOW_INDEX must be between 0 and {len(X) - 1} for {npz_path}")

    return {
        "window": X[window_index],
        "label": int(y[window_index]),
        "start_seconds": float(data["start_seconds"][window_index]),
        "end_seconds": float(data["end_seconds"][window_index]),
        "subject": scalar_string(data["subject"]),
        "bvp_hz": int(data["bvp_hz"]) if "bvp_hz" in data else BVP_HZ,
        "window_function": scalar_string(data["window_function"])
        if "window_function" in data
        else "unknown",
    }


def main():
    rect = load_window(Path(RECT_NPZ_PATH), WINDOW_INDEX)
    hanning = load_window(Path(HANNING_NPZ_PATH), WINDOW_INDEX)

    if rect["subject"] != hanning["subject"]:
        raise ValueError(f"subject mismatch: {rect['subject']} vs {hanning['subject']}")
    if rect["label"] != hanning["label"]:
        raise ValueError(f"label mismatch: {rect['label']} vs {hanning['label']}")
    if len(rect["window"]) != len(hanning["window"]):
        raise ValueError("window length mismatch")
    if not np.isclose(rect["start_seconds"], hanning["start_seconds"]) or not np.isclose(
        rect["end_seconds"], hanning["end_seconds"]
    ):
        raise ValueError("window time range mismatch")

    bvp_hz = rect["bvp_hz"]
    time = np.arange(len(rect["window"])) / bvp_hz
    label_name = LABEL_NAMES.get(rect["label"], "unknown")

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(
        f"{rect['subject']} window {WINDOW_INDEX} | {rect['label']}: {label_name} | "
        f"{rect['start_seconds']:.1f}s - {rect['end_seconds']:.1f}s"
    )

    axes[0].plot(time, rect["window"], color="#1f77b4", linewidth=0.8)
    axes[0].set_title("Rectangular window")
    axes[0].set_ylabel("BVP")
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, hanning["window"], color="#d62728", linewidth=0.8)
    axes[1].set_title("Hanning window")
    axes[1].set_ylabel("BVP")
    axes[1].grid(alpha=0.25)

    axes[2].plot(time, rect["window"], color="#1f77b4", linewidth=0.6, alpha=0.75, label="rectangular")
    axes[2].plot(time, hanning["window"], color="#d62728", linewidth=0.8, alpha=0.85, label="hanning")
    axes[2].set_title("Overlay")
    axes[2].set_xlabel("Window time (seconds)")
    axes[2].set_ylabel("BVP")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")

    axes[-1].set_xlim(0, len(rect["window"]) / bvp_hz)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
