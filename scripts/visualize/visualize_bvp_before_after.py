import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUBJECT = "S2"
BEFORE_PKL_PATH = rf"C:\ppg_stress\data\labeled\{SUBJECT}\{SUBJECT}.pkl"
AFTER_PKL_PATH = rf"C:\ppg_stress\data\preprocessed\all_preprocessed\{SUBJECT}\{SUBJECT}.pkl"
START_SECONDS = 0
END_SECONDS = None

BVP_HZ = 64


def load_bvp(pkl_path):
    with Path(pkl_path).open("rb") as file:
        data = pickle.load(file, encoding="latin1")
    return data.get("subject", Path(pkl_path).stem), np.asarray(
        data["signal"]["wrist"]["BVP"]
    ).reshape(-1)


def validate_time_range(start_seconds, end_seconds, duration_seconds):
    start = max(float(start_seconds), 0.0)
    end = duration_seconds if end_seconds is None else min(float(end_seconds), duration_seconds)
    if end <= start:
        raise ValueError("END_SECONDS must be greater than START_SECONDS")
    return start, end


def main():
    subject_before, bvp_before = load_bvp(BEFORE_PKL_PATH)
    subject_after, bvp_after = load_bvp(AFTER_PKL_PATH)

    if subject_before != subject_after:
        raise ValueError(f"subject mismatch: {subject_before} vs {subject_after}")
    if len(bvp_before) != len(bvp_after):
        raise ValueError(
            f"BVP length mismatch: before={len(bvp_before)}, after={len(bvp_after)}"
        )

    duration_seconds = len(bvp_before) / BVP_HZ
    start_seconds, end_seconds = validate_time_range(
        START_SECONDS,
        END_SECONDS,
        duration_seconds,
    )
    start_idx = int(start_seconds * BVP_HZ)
    end_idx = int(end_seconds * BVP_HZ)
    time = np.arange(start_idx, end_idx) / BVP_HZ

    before_segment = bvp_before[start_idx:end_idx]
    after_segment = bvp_after[start_idx:end_idx]

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(
        f"{subject_before} BVP before vs after preprocessing "
        f"({duration_seconds:.0f}s total, showing {start_seconds:.0f}-{end_seconds:.0f}s)"
    )

    axes[0].plot(time, before_segment, color="#1f77b4", linewidth=0.8)
    axes[0].set_title("Before preprocessing")
    axes[0].set_ylabel("BVP")
    axes[0].grid(alpha=0.25)

    axes[1].plot(time, after_segment, color="#d62728", linewidth=0.8)
    axes[1].set_title("After preprocessing")
    axes[1].set_ylabel("BVP")
    axes[1].grid(alpha=0.25)

    axes[2].plot(time, before_segment, color="#1f77b4", linewidth=0.6, alpha=0.75, label="before")
    axes[2].plot(time, after_segment, color="#d62728", linewidth=0.8, alpha=0.85, label="after")
    axes[2].set_title("Overlay")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("BVP")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.set_xlim(start_seconds, end_seconds)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
