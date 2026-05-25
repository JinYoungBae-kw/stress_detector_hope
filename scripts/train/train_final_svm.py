import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "3.5_peak_corrected"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"

STRESS_LABEL = 0
NONSTRESS_LABEL = 1
RANDOM_SEED = 42
CLASS_WEIGHT = None
OVERWRITE = True


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def load_subject_feature_file(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int8)
    feature_names = [str(name) for name in data["feature_names"]]
    subject = str(np.asarray(data["subject"]).item()) if "subject" in data else npz_path.stem

    finite_rows = np.all(np.isfinite(X), axis=1)
    if "nan_rows" in data:
        finite_rows = finite_rows & ~np.asarray(data["nan_rows"], dtype=bool)

    return {
        "subject": subject,
        "X": X[finite_rows],
        "y": y[finite_rows],
        "feature_names": feature_names,
        "removed_rows": int(np.sum(~finite_rows)),
        "source": str(npz_path),
        "samples_before": int(len(y)),
        "samples_after": int(np.sum(finite_rows)),
    }


def build_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight=CLASS_WEIGHT,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    np.random.seed(RANDOM_SEED)

    if OUTPUT_DIR.exists() and not OVERWRITE:
        existing = list(OUTPUT_DIR.glob("*"))
        if existing:
            raise FileExistsError(f"output directory already contains files: {OUTPUT_DIR}")

    npz_paths = sorted(FEATURE_DIR.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no feature NPZ files found in {FEATURE_DIR}")

    subjects = [load_subject_feature_file(path) for path in npz_paths]
    feature_names = subjects[0]["feature_names"]
    for subject_data in subjects:
        if subject_data["feature_names"] != feature_names:
            raise ValueError(f"feature name mismatch in {subject_data['source']}")
        if subject_data["X"].shape[1] != len(feature_names):
            raise ValueError(f"feature shape mismatch in {subject_data['source']}")

    X = np.vstack([subject_data["X"] for subject_data in subjects])
    y = np.concatenate([subject_data["y"] for subject_data in subjects])

    model = build_model()
    model.fit(X, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "svm_pipeline.joblib"
    joblib.dump(model, model_path)

    subject_rows = []
    for subject_data in subjects:
        subject_y = subject_data["y"]
        subject_rows.append(
            {
                "subject": subject_data["subject"],
                "source": subject_data["source"],
                "samples_before": subject_data["samples_before"],
                "samples_after": subject_data["samples_after"],
                "removed_rows": subject_data["removed_rows"],
                "stress_samples": int(np.sum(subject_y == STRESS_LABEL)),
                "nonstress_samples": int(np.sum(subject_y == NONSTRESS_LABEL)),
            }
        )

    write_csv(
        OUTPUT_DIR / "training_subjects.csv",
        subject_rows,
        [
            "subject",
            "source",
            "samples_before",
            "samples_after",
            "removed_rows",
            "stress_samples",
            "nonstress_samples",
        ],
    )

    config = {
        "feature_dir": str(FEATURE_DIR),
        "model_path": str(model_path),
        "feature_names": feature_names,
        "stress_label": STRESS_LABEL,
        "nonstress_label": NONSTRESS_LABEL,
        "random_seed": RANDOM_SEED,
        "model": {
            "pipeline": "StandardScaler + SVC",
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": CLASS_WEIGHT,
        },
        "training_samples": int(len(y)),
        "stress_samples": int(np.sum(y == STRESS_LABEL)),
        "nonstress_samples": int(np.sum(y == NONSTRESS_LABEL)),
        "subjects": [subject_data["subject"] for subject_data in subjects],
        "expected_pipeline": {
            "bvp_source": "wrist BVP",
            "bandpass_hz": [0.5, 3.5],
            "window_seconds": 360,
            "stride_seconds": 30,
            "window_function": "hanning",
            "peak_correction": "short NN interval correction",
            "nn_interval_valid_range_sec": [0.3, 2.0],
        },
    }

    with (OUTPUT_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    np.savez_compressed(
        OUTPUT_DIR / "metadata.npz",
        feature_names=np.asarray(feature_names),
        class_weight=np.asarray("none" if CLASS_WEIGHT is None else CLASS_WEIGHT),
        random_seed=np.asarray(RANDOM_SEED),
        stress_label=np.asarray(STRESS_LABEL),
        nonstress_label=np.asarray(NONSTRESS_LABEL),
        subjects=np.asarray([subject_data["subject"] for subject_data in subjects]),
        training_samples=np.asarray(len(y)),
        stress_samples=np.asarray(int(np.sum(y == STRESS_LABEL))),
        nonstress_samples=np.asarray(int(np.sum(y == NONSTRESS_LABEL))),
    )

    print(f"feature_dir: {FEATURE_DIR}")
    print(f"output_dir: {OUTPUT_DIR}")
    print(f"model saved: {model_path}")
    print(f"subjects: {len(subjects)}")
    print(f"training samples: {len(y)}")
    print(f"stress samples: {int(np.sum(y == STRESS_LABEL))}")
    print(f"non-stress samples: {int(np.sum(y == NONSTRESS_LABEL))}")
    print(f"features: {', '.join(feature_names)}")


if __name__ == "__main__":
    main()
