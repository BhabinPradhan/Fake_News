# verify_weights.py
import sys, os
sys.path.append(os.path.join(os.getcwd(), "MoPeD"))
from benchmark_loader import load_cases_from_csv
from model_manager import ModelManager

for run in range(2):
    manager = ModelManager()
    cases = load_cases_from_csv(
        "multimodal_dataset/test_final.csv",
        image_root="multimodal_dataset",
        n=60, use_images=True,
    )
    calib_cases = [c for c in cases if c.get("expected")][:30]
    manager.fit_label_order_from_benchmark(
        calib_cases, min_cases=10, min_improvement=0.20, auto_apply=True
    )
    manager.compute_reliability_from_benchmark(calib_cases)
    print(f"\n=== RUN {run+1} WEIGHTS ===")
    for k, v in manager.expert_reliability.items():
        print(f"  {k:<25} {v:.2f}")