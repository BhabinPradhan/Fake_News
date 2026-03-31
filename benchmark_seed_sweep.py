# benchmark_seed_sweep.py
# Run the same benchmark size across several random seeds and compare the results.
# This helps check whether the ensemble is stable or just doing well on one lucky sample.

import argparse
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "MoPeD"))

from benchmark_loader import load_cases_from_multiple_csvs
from model_manager import ModelManager


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


def parse_seeds(seed_text):
    seeds = []
    parts = seed_text.split(",")

    for part in parts:
        part = part.strip()
        if part:
            seeds.append(int(part))

    if not seeds:
        raise ValueError("At least one seed is required.")

    return seeds


def main():
    parser = argparse.ArgumentParser(
        description="Run the multimodal benchmark on several random seeds."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to one CSV file. If not found, all CSVs in multimodal_dataset are used.",
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
        default=60,
        help="Number of cases per run. The script keeps the sample balanced.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,7,13,21,42",
        help="Comma-separated random seeds.",
    )
    args = parser.parse_args()

    if args.csv:
        csv_paths = [args.csv]
    else:
        default_dir = os.path.join(SCRIPT_DIR, "multimodal_dataset")
        if os.path.isdir(default_dir):
            csv_paths = []
            for name in os.listdir(default_dir):
                if name.endswith(".csv"):
                    csv_paths.append(os.path.join(default_dir, name))
        else:
            csv_paths = []

    if not csv_paths:
        print("No CSVs found. Pass --csv path/to/file.csv")
        sys.exit(1)

    seeds = parse_seeds(args.seeds)
    n_per_csv = max(args.n // len(csv_paths), 4)

    print("Initializing ModelManager once for all runs...")
    manager = ModelManager()

    summaries = []

    print("\nPer-seed results")
    print("-" * 88)
    print(
        f"{'Seed':>6} {'Correct/Decided':>18} {'DecAcc':>8} {'Coverage':>9} "
        f"{'Uncertain':>10} {'RealAcc':>8} {'FakeAcc':>8}"
    )
    print("-" * 88)

    for seed in seeds:
        cases = load_cases_from_multiple_csvs(
            csv_paths,
            image_root=args.image_root,
            n_per_csv=n_per_csv,
            seed=seed,
        )

        if not cases:
            print(f"Seed {seed}: no valid cases loaded")
            continue

        results = manager.benchmark_prompts(cases)
        summary = summarize_results(results)
        summary["seed"] = seed
        summaries.append(summary)

        print(
            f"{seed:>6} {summary['correct']:>7}/{summary['decided']:<10} "
            f"{summary['decided_accuracy']:>7.1%} {summary['coverage']:>8.1%} "
            f"{summary['uncertain']:>10} {summary['real_accuracy']:>7.1%} "
            f"{summary['fake_accuracy']:>7.1%}"
        )

    if not summaries:
        print("\nNo successful runs.")
        sys.exit(1)

    decided_accuracies = [row["decided_accuracy"] for row in summaries]
    coverage_values = [row["coverage"] for row in summaries]
    uncertain_rates = [row["uncertain_rate"] for row in summaries]
    real_accuracies = [row["real_accuracy"] for row in summaries if row["real_decided"] > 0]
    fake_accuracies = [row["fake_accuracy"] for row in summaries if row["fake_decided"] > 0]

    print("\nAggregate summary")
    print("-" * 60)
    print(f"Seeds run:              {len(summaries)}")
    print(f"Mean decided accuracy:  {safe_mean(decided_accuracies):.1%}")
    print(f"Std decided accuracy:   {safe_std(decided_accuracies):.1%}")
    print(f"Min decided accuracy:   {min(decided_accuracies):.1%}")
    print(f"Max decided accuracy:   {max(decided_accuracies):.1%}")
    print(f"Mean coverage:          {safe_mean(coverage_values):.1%}")
    print(f"Mean uncertain rate:    {safe_mean(uncertain_rates):.1%}")
    print(f"Mean Real accuracy:     {safe_mean(real_accuracies):.1%}")
    print(f"Mean Fake accuracy:     {safe_mean(fake_accuracies):.1%}")

    print("\nRecommendation")
    print("-" * 60)
    print("Freeze the weights only after they look stable across several seeds.")
    print("A good final report should include both decided-case accuracy and coverage.")
    print("Do not re-fit weights on this same benchmark if you want it to stay a fair final check.")


if __name__ == "__main__":
    main()
