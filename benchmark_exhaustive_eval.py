# benchmark_exhaustive_eval.py
# Run the ensemble across several benchmark sizes and several seeds,
# then save both the detailed runs and the final summary to CSV files.

import argparse
import csv
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "MoPeD"))

from benchmark_loader import load_cases_from_multiple_csvs
from model_manager import ModelManager


def parse_int_list(text):
    values = []
    parts = text.split(",")

    for part in parts:
        part = part.strip()
        if part:
            values.append(int(part))

    if not values:
        raise ValueError("At least one integer value is required.")

    return values


def safe_mean(values):
    if values:
        return statistics.mean(values)
    return 0.0


def safe_std(values):
    if len(values) > 1:
        return statistics.pstdev(values)
    return 0.0


def summarize_results(results):
    total = len(results)
    decided_rows = [row for row in results if row["predicted"] != "Uncertain"]
    uncertain_rows = [row for row in results if row["predicted"] == "Uncertain"]
    correct_rows = [row for row in decided_rows if row.get("correct") is True]

    def class_accuracy(label_name):
        class_rows = [row for row in decided_rows if row["expected"] == label_name]
        if not class_rows:
            return 0.0, 0

        class_correct = sum(1 for row in class_rows if row.get("correct") is True)
        return class_correct / len(class_rows), len(class_rows)

    real_accuracy, real_decided = class_accuracy("Real")
    fake_accuracy, fake_decided = class_accuracy("Fake")

    summary = {
        "total": total,
        "decided": len(decided_rows),
        "correct": len(correct_rows),
        "coverage": len(decided_rows) / total if total else 0.0,
        "uncertain": len(uncertain_rows),
        "uncertain_rate": len(uncertain_rows) / total if total else 0.0,
        "real_accuracy": real_accuracy,
        "real_decided": real_decided,
        "fake_accuracy": fake_accuracy,
        "fake_decided": fake_decided,
    }

    if decided_rows:
        summary["decided_accuracy"] = len(correct_rows) / len(decided_rows)
    else:
        summary["decided_accuracy"] = 0.0

    return summary


def build_csv_paths(csv_arg):
    if csv_arg:
        return [csv_arg]

    dataset_dir = os.path.join(SCRIPT_DIR, "multimodal_dataset")
    csv_paths = []

    if os.path.isdir(dataset_dir):
        for name in os.listdir(dataset_dir):
            if name.endswith(".csv"):
                csv_paths.append(os.path.join(dataset_dir, name))

    return csv_paths


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run a larger benchmark sweep across several sizes and seeds."
    )
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
        "--n_values",
        type=str,
        default="60,80,100",
        help="Comma-separated benchmark sizes to test.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,3,5,7,9,11,13,15,21,42",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="benchmark_eval",
        help="Prefix for the output CSV files.",
    )
    args = parser.parse_args()

    csv_paths = build_csv_paths(args.csv)
    if not csv_paths:
        print("No CSVs found. Pass --csv path/to/file.csv")
        sys.exit(1)

    n_values = parse_int_list(args.n_values)
    seeds = parse_int_list(args.seeds)

    print("Initializing ModelManager once for the full sweep...")
    manager = ModelManager()

    per_run_rows = []

    print("\nPer-run results")
    print("-" * 100)
    print(
        f"{'n':>6} {'seed':>6} {'correct/decided':>18} {'dec_acc':>9} "
        f"{'coverage':>9} {'uncertain':>10} {'real_acc':>9} {'fake_acc':>9}"
    )
    print("-" * 100)

    for n in n_values:
        n_per_csv = max(n // len(csv_paths), 4)

        for seed in seeds:
            cases = load_cases_from_multiple_csvs(
                csv_paths,
                image_root=args.image_root,
                n_per_csv=n_per_csv,
                seed=seed,
            )

            if not cases:
                continue

            results = manager.benchmark_prompts(cases)
            summary = summarize_results(results)

            row = {
                "n": n,
                "seed": seed,
                "total": summary["total"],
                "decided": summary["decided"],
                "correct": summary["correct"],
                "decided_accuracy": summary["decided_accuracy"],
                "coverage": summary["coverage"],
                "uncertain": summary["uncertain"],
                "uncertain_rate": summary["uncertain_rate"],
                "real_accuracy": summary["real_accuracy"],
                "real_decided": summary["real_decided"],
                "fake_accuracy": summary["fake_accuracy"],
                "fake_decided": summary["fake_decided"],
            }
            per_run_rows.append(row)

            print(
                f"{n:>6} {seed:>6} {row['correct']:>7}/{row['decided']:<10} "
                f"{row['decided_accuracy']:>8.1%} {row['coverage']:>8.1%} "
                f"{row['uncertain']:>10} {row['real_accuracy']:>8.1%} "
                f"{row['fake_accuracy']:>8.1%}"
            )

    if not per_run_rows:
        print("\nNo successful runs.")
        sys.exit(1)

    aggregate_rows = []

    print("\nAggregate by benchmark size")
    print("-" * 100)
    print(
        f"{'n':>6} {'runs':>6} {'mean_acc':>10} {'std_acc':>9} {'min_acc':>9} "
        f"{'max_acc':>9} {'mean_cov':>10} {'mean_unc':>10}"
    )
    print("-" * 100)

    for n in n_values:
        rows_for_n = [row for row in per_run_rows if row["n"] == n]
        if not rows_for_n:
            continue

        decided_accuracies = [row["decided_accuracy"] for row in rows_for_n]
        coverage_values = [row["coverage"] for row in rows_for_n]
        uncertain_rates = [row["uncertain_rate"] for row in rows_for_n]
        real_accuracies = [row["real_accuracy"] for row in rows_for_n if row["real_decided"] > 0]
        fake_accuracies = [row["fake_accuracy"] for row in rows_for_n if row["fake_decided"] > 0]

        aggregate_row = {
            "n": n,
            "runs": len(rows_for_n),
            "mean_decided_accuracy": safe_mean(decided_accuracies),
            "std_decided_accuracy": safe_std(decided_accuracies),
            "min_decided_accuracy": min(decided_accuracies),
            "max_decided_accuracy": max(decided_accuracies),
            "mean_coverage": safe_mean(coverage_values),
            "std_coverage": safe_std(coverage_values),
            "mean_uncertain_rate": safe_mean(uncertain_rates),
            "mean_real_accuracy": safe_mean(real_accuracies),
            "mean_fake_accuracy": safe_mean(fake_accuracies),
        }
        aggregate_rows.append(aggregate_row)

        print(
            f"{n:>6} {aggregate_row['runs']:>6} {aggregate_row['mean_decided_accuracy']:>9.1%} "
            f"{aggregate_row['std_decided_accuracy']:>8.1%} {aggregate_row['min_decided_accuracy']:>8.1%} "
            f"{aggregate_row['max_decided_accuracy']:>8.1%} {aggregate_row['mean_coverage']:>9.1%} "
            f"{aggregate_row['mean_uncertain_rate']:>9.1%}"
        )

    per_run_path = os.path.join(SCRIPT_DIR, f"{args.output_prefix}_per_run.csv")
    aggregate_path = os.path.join(SCRIPT_DIR, f"{args.output_prefix}_aggregate.csv")

    per_run_fields = [
        "n",
        "seed",
        "total",
        "decided",
        "correct",
        "decided_accuracy",
        "coverage",
        "uncertain",
        "uncertain_rate",
        "real_accuracy",
        "real_decided",
        "fake_accuracy",
        "fake_decided",
    ]

    aggregate_fields = [
        "n",
        "runs",
        "mean_decided_accuracy",
        "std_decided_accuracy",
        "min_decided_accuracy",
        "max_decided_accuracy",
        "mean_coverage",
        "std_coverage",
        "mean_uncertain_rate",
        "mean_real_accuracy",
        "mean_fake_accuracy",
    ]

    write_csv(per_run_path, per_run_rows, per_run_fields)
    write_csv(aggregate_path, aggregate_rows, aggregate_fields)

    print("\nSaved files")
    print("-" * 100)
    print(f"Per-run CSV:     {per_run_path}")
    print(f"Aggregate CSV:   {aggregate_path}")

    print("\nHow to use this")
    print("-" * 100)
    print("If one weight setting keeps the accuracy higher without hurting coverage too much,")
    print("that is a good sign that the weights are ready to freeze.")
    print("Try not to keep tuning against the same CSV once you already have enough evidence.")


if __name__ == "__main__":
    main()
