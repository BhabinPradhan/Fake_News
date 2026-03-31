# benchmark_loader.py
# Load benchmark cases from one or more CSV files and run them through ModelManager.
# This version is multimodal only, so every case must have both text and a real image.

import os
import sys
import argparse
import random
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "multimodel_dataset")

if os.path.isdir(DATASET_DIR):
    DEFAULT_CSVS = []
    for name in os.listdir(DATASET_DIR):
        if name.endswith(".csv"):
            DEFAULT_CSVS.append(os.path.join(DATASET_DIR, name))
else:
    DEFAULT_CSVS = []

sys.path.append(os.path.join(SCRIPT_DIR, "MoPeD"))
from model_manager import ModelManager


def load_cases_from_csv(csv_path, image_root=None, n=20, seed=42):
    # Read one CSV and return a balanced list of benchmark cases.
    # det_fake_label = 1 means Fake, and 0 means Real.
    if image_root is None:
        image_root = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)

    # Be a bit flexible with column names.
    df.columns = [column.strip().lower() for column in df.columns]
    text_col = next((column for column in df.columns if column in ("content", "text", "caption", "title")), None)
    image_col = next((column for column in df.columns if column in ("image", "image_path", "img")), None)
    label_col = next((column for column in df.columns if "label" in column or "fake" in column), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not find text/label columns in {csv_path}. Found: {list(df.columns)}"
        )

    if image_col is None:
        raise ValueError(
            f"Could not find an image column in {csv_path}. Found: {list(df.columns)}"
        )

    # Keep only rows with usable text.
    df = df.dropna(subset=[text_col])
    df["_text"] = df[text_col].astype(str).str.strip()
    df["_label"] = df[label_col].apply(lambda value: "Fake" if int(value) == 1 else "Real")

    # Build full image paths and keep only rows where the image actually exists.
    df["_img"] = df[image_col].apply(
        lambda value: os.path.join(image_root, str(value)) if pd.notna(value) else None
    )
    df = df[df["_img"].apply(lambda path: path is not None and os.path.exists(path))]

    if len(df) == 0:
        raise ValueError(f"No matching image files found under {image_root}.")

    # Sample an equal number of Real and Fake rows.
    half = n // 2

    fake_rows = df[df["_label"] == "Fake"]
    real_rows = df[df["_label"] == "Real"]

    sampled_fakes = fake_rows.sample(min(half, len(fake_rows)), random_state=seed)
    sampled_reals = real_rows.sample(min(half, len(real_rows)), random_state=seed)
    sampled = pd.concat([sampled_fakes, sampled_reals]).sample(frac=1, random_state=seed)

    cases = []
    for _, row in sampled.iterrows():
        case = {
            "text": row["_text"],
            "expected": row["_label"],
            "image_path": row["_img"],
        }
        cases.append(case)

    print(f"  Loaded {len(cases)} cases from {os.path.basename(csv_path)} (with images)")
    print(
        f"    Fake: {sum(1 for case in cases if case['expected'] == 'Fake')}  "
        f"Real: {sum(1 for case in cases if case['expected'] == 'Real')}"
    )
    return cases


def load_cases_from_multiple_csvs(csv_paths, image_root=None, n_per_csv=10, seed=42):
    # Load cases from several CSV files and shuffle them together.
    all_cases = []

    for path in csv_paths:
        try:
            cases = load_cases_from_csv(path, image_root, n_per_csv, seed)
            all_cases.extend(cases)
        except Exception as error:
            print(f"  Skipping {path}: {error}")

    random.seed(seed)
    random.shuffle(all_cases)
    return all_cases


def print_results(results):
    # Print a short benchmark summary in the terminal.
    correct = sum(1 for row in results if row.get("correct") is True)
    uncertain = sum(1 for row in results if row["predicted"] == "Uncertain")
    total = len(results)
    decided = total - uncertain

    print(f"\n{'=' * 60}")
    print(
        f"BENCHMARK  {total} cases | "
        f"Correct: {correct}/{decided} ({100 * correct / max(decided, 1):.0f}%) | "
        f"Uncertain: {uncertain}"
    )
    print(f"{'=' * 60}")

    for row in results:
        if row.get("correct"):
            icon = "OK"
        elif row["predicted"] == "Uncertain":
            icon = "??"
        else:
            icon = "XX"

        snippet = row["text"][:55].replace("\n", " ")
        print(
            f"  {icon} [IMG] {row['expected']:4s}->{row['predicted']:9s} "
            f"conf={row['confidence']:.0%} | {snippet}"
        )

    print(f"{'=' * 60}")
    for label_name in ("Real", "Fake"):
        subset = [
            row for row in results
            if row["expected"] == label_name and row["predicted"] != "Uncertain"
        ]
        if subset:
            accuracy = sum(1 for row in subset if row.get("correct")) / len(subset)
            print(f"  {label_name:4s} accuracy: {accuracy:.0%} ({len(subset)} decided)")


def main():
    parser = argparse.ArgumentParser(description="Run ModelManager benchmark from CSV datasets")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to one CSV file. If omitted, all CSVs in multimodal_dataset are used.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Base directory for image files in the CSV.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of cases to sample. The script keeps the sample balanced.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run label-order and reliability calibration before the benchmark.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Decide which CSV files to use.
    if args.csv:
        csv_paths = [args.csv]
    elif DEFAULT_CSVS:
        csv_paths = DEFAULT_CSVS
        print(f"Found {len(csv_paths)} CSV(s) in multimodal_dataset/:")
        for path in csv_paths:
            print(f"  {os.path.basename(path)}")
    else:
        print("No CSVs found. Pass --csv path/to/file.csv")
        sys.exit(1)

    print("\nLoading test cases...")
    n_per_csv = max(args.n // len(csv_paths), 4)
    cases = load_cases_from_multiple_csvs(
        csv_paths,
        image_root=args.image_root,
        n_per_csv=n_per_csv,
        seed=args.seed,
    )

    if not cases:
        print("No valid cases loaded. Check your CSV paths and image_root.")
        sys.exit(1)

    print("\nInitializing ModelManager...")
    manager = ModelManager()

    # Optional calibration step before the final benchmark.
    if args.calibrate:
        calib_cases = [case for case in cases if case.get("expected")][:30]

        if len(calib_cases) >= 10:
            print("\nRunning label-order calibration on first 30 labeled cases...")
            report = manager.fit_label_order_from_benchmark(
                calib_cases,
                min_cases=10,
                min_improvement=0.20,
                auto_apply=True,
            )

            print(f"{'Model':<25} {'Order':>6}  {'Normal':>7}  {'Flipped':>8}")
            print("-" * 55)

            for model, stats in report.items():
                flag = " <- FLIPPED" if stats["applied_order"] == "FR" else ""
                print(
                    f"  {model:<23} {stats['applied_order']:>6}  "
                    f"{stats['normal_acc']:>6.0%}   {stats['flipped_acc']:>7.0%}{flag}"
                )

            print("\nComputing reliability weights from calibration cases...")
            manager.compute_reliability_from_benchmark(calib_cases)

            print("Computing short-text reliability weights from calibration cases...")
            try:
                manager.compute_short_text_reliability_from_benchmark(
                    calib_cases,
                    max_words=manager.short_text_max_words,
                    min_cases=10,
                )
            except ValueError as error:
                print(f"Skipping short-text reliability calibration: {error}")
        else:
            print("Not enough labeled cases for calibration, skipping.")

    print(f"\nRunning benchmark on {len(cases)} cases...")
    results = manager.benchmark_prompts(cases)
    print_results(results)


if __name__ == "__main__":
    main()
