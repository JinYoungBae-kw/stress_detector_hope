import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BVP_HZ = 64


def load_bvp(pkl_path):
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")

    subject = data.get("subject", pkl_path.stem)
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=np.float64).reshape(-1)
    return subject, bvp


def validate_time_range(start_seconds, end_seconds, duration_seconds):
    start = max(float(start_seconds), 0.0)
    end = duration_seconds if end_seconds is None else min(float(end_seconds), duration_seconds)
    if end <= start:
        raise ValueError("END_SECONDS must be greater than START_SECONDS")
    return start, end


def plot_bvp_stage_comparison(
    before_pkl_path,
    after_pkl_path,
    stage_name,
    start_seconds=0,
    end_seconds=None,
):
    before_subject, before_bvp = load_bvp(before_pkl_path)
    after_subject, after_bvp = load_bvp(after_pkl_path)

    if before_subject != after_subject:
        raise ValueError(f"subject mismatch: {before_subject} vs {after_subject}")

    if len(before_bvp) != len(after_bvp):
        raise ValueError(
            f"BVP length mismatch: before={len(before_bvp)}, after={len(after_bvp)}"
        )

    duration_seconds = len(before_bvp) / BVP_HZ
    start_seconds, end_seconds = validate_time_range(
        start_seconds,
        end_seconds,
        duration_seconds,
    )
    start_idx = int(start_seconds * BVP_HZ)
    end_idx = int(end_seconds * BVP_HZ)
    time = np.arange(start_idx, end_idx) / BVP_HZ

    before_segment = before_bvp[start_idx:end_idx]
    after_segment = after_bvp[start_idx:end_idx]

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    fig.suptitle(
        f"{before_subject} BVP: {stage_name} "
        f"({duration_seconds:.0f}s total, showing {start_seconds:.0f}-{end_seconds:.0f}s)"
    )

    axes[0].plot(time, after_segment, color="#d62728", linewidth=0.8)
    axes[0].set_title(f"BVP after {stage_name}")
    axes[0].set_ylabel("BVP")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        time,
        before_segment,
        color="#1f77b4",
        linewidth=0.6,
        alpha=0.75,
        label="original",
    )
    axes[1].plot(
        time,
        after_segment,
        color="#d62728",
        linewidth=0.8,
        alpha=0.85,
        label=stage_name,
    )
    axes[1].set_title("Overlay: original vs preprocessed")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("BVP")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    for ax in axes:
        ax.set_xlim(start_seconds, end_seconds)

    fig.tight_layout()
    plt.show()
