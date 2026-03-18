"""
benchmark_loader.py
-------------------
Loads ground-truth test cases from your multimodal CSV datasets and runs
them through ModelManager.  Supports both text-only and text+image modes.

Usage (from BERT_Project/):
    python benchmark_loader.py                        # default: 20 balanced cases
    python benchmark_loader.py --n 50 --image         # 50 cases with real images
    python benchmark_loader.py --csv path/to/file.csv # custom CSV
    python benchmark_loader.py --calibrate            # run label-order fitter first
"""

import os
import sys
import argparse
import random
import pandas as pd

# Handle the paths and imports for ModelManager
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(SCRIPT_DIR, "multimodel_dataset")
DEFAULT_CSVS = [
    os.path.join(DATASET_DIR, f)
    for f in os.listdir(DATASET_DIR)
    if f.endswith(".csv")
] if os.path.isdir(DATASET_DIR) else []

sys.path.append(os.path.join(SCRIPT_DIR, "MoPeD"))
from model_manager import ModelManager


# Load the CSV and prepare benchmark cases for ModelManager

def load_cases_from_csv(
    csv_path: str,
    image_root: str = None,
    n: int = 20,
    use_images: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    Reads a CSV with columns: content, image, det_fake_label
    Returns a balanced list of benchmark cases ready for ModelManager.

    det_fake_label: 1 = Fake, 0 = Real  (your CSV convention)

    Args:
        csv_path:   path to the CSV file
        image_root: base directory for image paths in the CSV.
                    If None, uses the CSV's directory.
        n:          total number of cases to sample (balanced Real/Fake)
        use_images: if True, include image paths; if False, text-only
        seed:       random seed for reproducible sampling
    """
    if image_root is None:
        image_root = os.path.dirname(csv_path)

    df = pd.read_csv(csv_path)

    # We need to be flexible about column names, so we normalize them and look for keywords
    df.columns = [c.strip().lower() for c in df.columns]
    text_col  = next((c for c in df.columns if c in ("content", "text", "caption", "title")), None)
    image_col = next((c for c in df.columns if c in ("image", "image_path", "img")), None)
    label_col = next((c for c in df.columns if "label" in c or "fake" in c), None)

    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns in {csv_path}. "
                         f"Found: {list(df.columns)}")

    # Drop rows with missing text
    df = df.dropna(subset=[text_col])
    df["_text"]  = df[text_col].astype(str).str.strip()
    df["_label"] = df[label_col].apply(lambda v: "Fake" if int(v) == 1 else "Real")

    # Resolve image paths
    if use_images and image_col:
        df["_img"] = df[image_col].apply(
            lambda p: os.path.join(image_root, str(p)) if pd.notna(p) else None
        )
        # Only keep rows where the image file actually exists
        df = df[df["_img"].apply(lambda p: p is not None and os.path.exists(p))]
        if len(df) == 0:
            print(f"  ⚠ No matching image files found under {image_root}. "
                  "Falling back to text-only mode.")
            use_images = False

    # Balanced sampling: equal Real and Fake
    half = n // 2
    fakes = df[df["_label"] == "Fake"].sample(min(half, len(df[df["_label"] == "Fake"])),
                                               random_state=seed)
    reals = df[df["_label"] == "Real"].sample(min(half, len(df[df["_label"] == "Real"])),
                                               random_state=seed)
    sampled = pd.concat([fakes, reals]).sample(frac=1, random_state=seed)  # shuffle

    cases = []
    for _, row in sampled.iterrows():
        case = {
            "text":     row["_text"],
            "expected": row["_label"],
        }
        if use_images and image_col:
            case["image_path"] = row["_img"]
        cases.append(case)

    print(f"  Loaded {len(cases)} cases from {os.path.basename(csv_path)} "
          f"({'with images' if use_images and image_col else 'text-only'})")
    print(f"    Fake: {sum(1 for c in cases if c['expected']=='Fake')}  "
          f"Real: {sum(1 for c in cases if c['expected']=='Real')}")
    return cases


def load_cases_from_multiple_csvs(
    csv_paths: list[str],
    image_root: str = None,
    n_per_csv: int = 10,
    use_images: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Load and combine cases from multiple CSV files."""
    all_cases = []
    for path in csv_paths:
        try:
            cases = load_cases_from_csv(path, image_root, n_per_csv, use_images, seed)
            all_cases.extend(cases)
        except Exception as e:
            print(f"  ⚠ Skipping {path}: {e}")
    random.seed(seed)
    random.shuffle(all_cases)
    return all_cases


# ── Pretty results printer ─────────────────────────────────────────────────

def print_results(results: list[dict]):
    correct   = sum(1 for r in results if r.get("correct") is True)
    incorrect = sum(1 for r in results if r.get("correct") is False)
    uncertain = sum(1 for r in results if r["predicted"] == "Uncertain")
    total     = len(results)
    decided   = total - uncertain

    print(f"\n{'='*60}")
    print(f"BENCHMARK  {total} cases | "
          f"Correct: {correct}/{decided} ({100*correct/max(decided,1):.0f}%) | "
          f"Uncertain: {uncertain}")
    print(f"{'='*60}")

    for r in results:
        icon     = "OK" if r.get("correct") else ("??" if r["predicted"] == "Uncertain" else "XX")
        src      = "IMG" if r.get("used_real_image") else "TXT"
        snippet  = r["text"][:55].replace("\n", " ")
        print(f"  {icon} [{src}] {r['expected']:4s}->{r['predicted']:9s} "
              f"conf={r['confidence']:.0%} | {snippet}")

    print(f"{'='*60}")
    for label in ("Real", "Fake"):
        subset = [r for r in results if r["expected"] == label
                  and r["predicted"] != "Uncertain"]
        if subset:
            acc = sum(1 for r in subset if r.get("correct")) / len(subset)
            print(f"  {label:4s} accuracy: {acc:.0%} ({len(subset)} decided)")


#  Handle the command-line interface and run the benchmark
def main():
    parser = argparse.ArgumentParser(description="Run ModelManager benchmark from CSV datasets")
    parser.add_argument("--csv",       type=str, default=None,
                        help="Path to a single CSV file (default: all CSVs in multimodel_dataset/)")
    parser.add_argument("--image_root",type=str, default=None,
                        help="Base directory for image files in the CSV")
    parser.add_argument("--n",         type=int, default=20,
                        help="Number of cases to sample (balanced, default 20)")
    parser.add_argument("--image",     action="store_true",
                        help="Use real images from the CSV (default: text-only)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run label-order fitter before benchmark (uses first 10 cases)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    # Load the cases from CSV(s)
    if args.csv:
        csv_paths = [args.csv]
    elif DEFAULT_CSVS:
        csv_paths = DEFAULT_CSVS
        print(f"Found {len(csv_paths)} CSV(s) in multimodel_dataset/:")
        for p in csv_paths:
            print(f"  {os.path.basename(p)}")
    else:
        print("No CSVs found. Pass --csv path/to/file.csv")
        sys.exit(1)

    print("\nLoading test cases...")
    n_per_csv = max(args.n // len(csv_paths), 4)
    cases = load_cases_from_multiple_csvs(
        csv_paths,
        image_root=args.image_root,
        n_per_csv=n_per_csv,
        use_images=args.image,
        seed=args.seed,
    )

    if not cases:
        print("No valid cases loaded. Check your CSV paths and image_root.")
        sys.exit(1)

    # The cases are now ready to be fed into ModelManager for benchmarking...
    print("\nInitializing ModelManager...")
    manager = ModelManager()

    # calibrate label order first 
    if args.calibrate:
        calib_cases = [c for c in cases if c.get("expected")][:30]
        if len(calib_cases) >= 10:
            print("\nRunning label-order calibration on first 30 labeled cases...")
            report = manager.fit_label_order_from_benchmark(
                calib_cases, min_cases=10, min_improvement=0.20, auto_apply=True
            )
            print(f"{'Model':<25} {'Order':>6}  {'Normal':>7}  {'Flipped':>8}")
            print("-" * 55)
            for model, stats in report.items():
                flag = " ← FLIPPED" if stats["applied_order"] == "FR" else ""
                print(f"  {model:<23} {stats['applied_order']:>6}  "
                      f"{stats['normal_acc']:>6.0%}   {stats['flipped_acc']:>7.0%}{flag}")
            print("\nComputing reliability weights from calibration cases...")
            manager.compute_reliability_from_benchmark(calib_cases)
            print("Computing short-text reliability weights from calibration cases...")
            try:
                manager.compute_short_text_reliability_from_benchmark(
                    calib_cases,
                    max_words=manager.short_text_max_words,
                    min_cases=10,
                )
            except ValueError as e:
                print(f"Skipping short-text reliability calibration: {e}")
        else:
            print("Not enough labeled cases for calibration, skipping.")

    # Run benchmark 
    print(f"\nRunning benchmark on {len(cases)} cases...")
    results = manager.benchmark_prompts(cases)
    print_results(results)


if __name__ == "__main__":
    main()
