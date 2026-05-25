from pathlib import Path

from _preprocess_compare_common import plot_bvp_stage_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBJECT = "S2"
BEFORE_PKL_PATH = PROJECT_ROOT / "data" / "labeled" / SUBJECT / f"{SUBJECT}.pkl"
AFTER_PKL_PATH = PROJECT_ROOT / "data" / "preprocessed" / "moving_average" / SUBJECT / f"{SUBJECT}.pkl"

START_SECONDS = 0
END_SECONDS = None


def main():
    plot_bvp_stage_comparison(
        before_pkl_path=BEFORE_PKL_PATH,
        after_pkl_path=AFTER_PKL_PATH,
        stage_name="bandpass + kalman + moving average",
        start_seconds=START_SECONDS,
        end_seconds=END_SECONDS,
    )


if __name__ == "__main__":
    main()
