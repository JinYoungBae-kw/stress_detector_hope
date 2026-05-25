import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PKL_PATH = r"C:\ppg_stress\data\preprocessed\all_preprocessed\S2\S2.pkl"
START_SECONDS = 0
END_SECONDS = None

STAGE_NAME = "all_preprocessed"
LABEL_HZ = 700
BVP_HZ = 64

LABEL_NAMES = {
    0: "stress",
    1: "non-stress",
    2: "ignore",
}

LABEL_COLORS = {
    0: "#d62728",
    1: "#2ca02c",
    2: "#9e9e9e",
}


def contiguous_segments(labels):
    change_points = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, len(labels)]
    return zip(starts, ends, labels[starts])


def validate_time_range(start_seconds, end_seconds, duration_seconds):
    start = max(float(start_seconds), 0.0)
    end = duration_seconds if end_seconds is None else min(float(end_seconds), duration_seconds)
    if end <= start:
        raise ValueError("END_SECONDS must be greater than START_SECONDS")
    return start, end


def make_legend(labels):
    return [
        Line2D(
            [0],
            [0],
            color=LABEL_COLORS.get(label, "#000000"),
            lw=4,
            label=f"{label}: {LABEL_NAMES.get(label, 'unknown')}",
        )
        for label in sorted(labels)
    ]


def plot_label(ax, labels, start_seconds, end_seconds):
    start_idx = int(start_seconds * LABEL_HZ)
    end_idx = int(end_seconds * LABEL_HZ)
    visible_labels = set()

    for seg_start, seg_end, label in contiguous_segments(labels):
        draw_start = max(seg_start, start_idx)
        draw_end = min(seg_end, end_idx)
        if draw_end <= draw_start:
            continue

        label = int(label)
        x = np.array([draw_start, draw_end - 1]) / LABEL_HZ
        y = np.array([label, label])
        ax.plot(
            x,
            y,
            color=LABEL_COLORS.get(label, "#000000"),
            linewidth=6,
            solid_capstyle="butt",
        )
        visible_labels.add(label)

    ax.set_ylabel("Label")
    ax.set_yticks(sorted(visible_labels) if visible_labels else [0, 1, 2])
    ax.set_ylim(-0.5, 2.5)
    ax.grid(axis="x", alpha=0.25)
    if visible_labels:
        ax.legend(handles=make_legend(visible_labels), loc="upper right")


def plot_colored_bvp(ax, bvp, label_bvp, start_seconds, end_seconds):
    start_idx = int(start_seconds * BVP_HZ)
    end_idx = int(end_seconds * BVP_HZ)
    visible_labels = set()

    for seg_start, seg_end, label in contiguous_segments(label_bvp):
        draw_start = max(seg_start, start_idx)
        draw_end = min(seg_end, end_idx)
        if draw_end <= draw_start:
            continue

        label = int(label)
        time = np.arange(draw_start, draw_end) / BVP_HZ
        ax.plot(
            time,
            bvp[draw_start:draw_end],
            color=LABEL_COLORS.get(label, "#000000"),
            linewidth=0.8,
        )
        visible_labels.add(label)

    ax.set_ylabel("BVP")
    ax.grid(alpha=0.25)
    if visible_labels:
        ax.legend(handles=make_legend(visible_labels), loc="upper right")


def plot_plain_bvp(ax, bvp, start_seconds, end_seconds):
    start_idx = int(start_seconds * BVP_HZ)
    end_idx = int(end_seconds * BVP_HZ)
    time = np.arange(start_idx, end_idx) / BVP_HZ

    ax.plot(time, bvp[start_idx:end_idx], color="#1f77b4", linewidth=0.8)
    ax.set_ylabel("BVP")
    ax.set_xlabel("Time (seconds)")
    ax.grid(alpha=0.25)


def main():
    pkl_path = Path(PKL_PATH)
    with pkl_path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")

    subject = data.get("subject", pkl_path.stem)
    labels = np.asarray(data["label"]).reshape(-1)
    bvp = np.asarray(data["signal"]["wrist"]["BVP"]).reshape(-1)

    if "label_bvp" not in data:
        raise KeyError("data['label_bvp'] was not found. Run preprocessing first.")

    label_bvp = np.asarray(data["label_bvp"]).reshape(-1)
    if len(label_bvp) != len(bvp):
        raise ValueError(
            f"label_bvp length ({len(label_bvp)}) does not match BVP length ({len(bvp)})."
        )

    label_duration = len(labels) / LABEL_HZ
    bvp_duration = len(bvp) / BVP_HZ
    if not np.isclose(label_duration, bvp_duration, atol=1 / BVP_HZ):
        raise ValueError(
            f"duration mismatch: label={label_duration:.6f}s, BVP={bvp_duration:.6f}s"
        )

    start_seconds, end_seconds = validate_time_range(
        START_SECONDS,
        END_SECONDS,
        bvp_duration,
    )

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(
        f"{subject} {STAGE_NAME} PKL check "
        f"({bvp_duration:.0f}s total, showing {start_seconds:.0f}-{end_seconds:.0f}s)"
    )

    plot_label(axes[0], labels, start_seconds, end_seconds)
    axes[0].set_title("Simplified label")

    plot_colored_bvp(axes[1], bvp, label_bvp, start_seconds, end_seconds)
    axes[1].set_title("BVP colored by label_bvp")

    plot_plain_bvp(axes[2], bvp, start_seconds, end_seconds)
    axes[2].set_title("Plain BVP")

    for ax in axes:
        ax.set_xlim(start_seconds, end_seconds)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
