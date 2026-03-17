# threshold_search.py
import sys, os
sys.path.append(os.path.join(os.getcwd(), "MoPeD"))

from benchmark_loader import load_cases_from_csv
from model_manager import ModelManager

manager = ModelManager()

cases = load_cases_from_csv(
    "multimodal_dataset/test_final.csv",
    image_root="multimodal_dataset",
    n=60,
    use_images=True,
)

# ── Run calibration first — same as --calibrate flag ──────────────────
calib_cases = [c for c in cases if c.get("expected")][:30]
print("Running label-order calibration...")
manager.fit_label_order_from_benchmark(
    calib_cases, min_cases=10, min_improvement=0.20, auto_apply=True
)

print("Computing reliability weights...")
manager.compute_reliability_from_benchmark(calib_cases)

# ── Now run grid search ────────────────────────────────────────────────
vote_strengths = [0.10, 0.15, 0.20, 0.25]
agreements     = [0.55, 0.60, 0.65, 0.70]

print(f"\n{'VS':>6} {'AGR':>6} {'Acc':>6} {'Unc':>6} {'Real':>6} {'Fake':>6}")
print("-" * 42)

best_score = 0
best_combo = None

for vs in vote_strengths:
    for agr in agreements:
        manager.min_vote_strength = vs
        manager.min_agreement     = agr

        results = manager.benchmark_prompts(cases)

        total     = len(results)
        uncertain = sum(1 for r in results if r["predicted"] == "Uncertain")
        decided   = total - uncertain
        correct   = sum(1 for r in results if r.get("correct") is True)

        real_sub  = [r for r in results if r["expected"] == "Real"  and r["predicted"] != "Uncertain"]
        fake_sub  = [r for r in results if r["expected"] == "Fake"  and r["predicted"] != "Uncertain"]
        real_acc  = sum(1 for r in real_sub if r.get("correct")) / max(len(real_sub), 1)
        fake_acc  = sum(1 for r in fake_sub if r.get("correct")) / max(len(fake_sub), 1)
        overall   = correct / max(decided, 1)

        score = fake_acc * 0.6 + overall * 0.4
        if score > best_score:
            best_score = score
            best_combo = (vs, agr)

        print(f"  {vs:>4.2f}  {agr:>4.2f}  {overall:>5.0%}  {uncertain:>5}  {real_acc:>5.0%}  {fake_acc:>5.0%}")

print(f"\nBest combo: vote_strength={best_combo[0]}, agreement={best_combo[1]}")