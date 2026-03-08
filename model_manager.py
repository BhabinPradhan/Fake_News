import os
import re
import sys
from PIL import Image
import torch
# Bridge the gap to the MoPeD subdirectory
sys.path.append(os.path.join(os.getcwd(), "MoPeD"))

# Import your verified wrappers
from moped_wrapper import MopedInference
from coolant_wrapper import CoolantInference
from emaf_wrapper import EmafInference

class ModelManager:
    def __init__(self):
        print("Initializing Global Model Manager...")
        
        # --- 1. Initialize MoPeD Experts ---
        self.moped_experts = {
            "xfacta": MopedInference("MoPeD/best_moped_xfacta.pth", dataset_type='english', device=torch.device('cuda:0')),
            "snopes": MopedInference("MoPeD/best_moped_snopes.pth", dataset_type='english', device=torch.device('cuda:0')),
            "weibo":  MopedInference("MoPeD/best_moped_weibo.pth", dataset_type='weibo', device=torch.device('cuda:0')),
            "mmhl":   MopedInference("MoPeD/best_moped_mmhl_fold0.pth", dataset_type='english', device=torch.device('cuda:1'))
        }
        self.coolant_experts = {
            "xfacta": CoolantInference("best_model_coolant_xfacta.pth", device=torch.device('cuda:1')),
            "snopes": CoolantInference("best_model_coolant_multimodal.pth", device=torch.device('cuda:1')),
            "weibo":  CoolantInference("best_coolant_weibo.pth", device=torch.device('cuda:2')),
            "mmhl":   CoolantInference("best_model_coolant_mmhl_fold4.pth", device=torch.device('cuda:2'))
        }
        self.emaf_experts = {
            "xfacta": EmafInference("best_emaf_xfacta.pth", lang='en', device=torch.device('cuda:2')),
            "snopes": EmafInference("best_emaf_multimodal.pth", lang='en', device=torch.device('cuda:3')),
            "weibo":  EmafInference("best_emaf_paper.pth", lang='zh', device=torch.device('cuda:3')),
            "mmhl":   EmafInference("best_emaf_mmhl_fold3.pth", lang='en', device=torch.device('cuda:3'))
        }
        # --- FUTURE: Add Teammate Experts Here ---
        # self.mcan_experts = { ... }

        # Per-expert reliability weights (option 3).
        # Keep these configurable and tune from your validation benchmarks.
        self.expert_reliability = {
            # Snopes variants trained on snopes+fakeddit — same domain as test set
            "MoPeD (snopes)":    1.00,
            "COOLANT (snopes)":  1.00,
            "EMAF (snopes)":     0.60,  # partially works — not fully collapsed

            # XFacta variants: collapsed toward Real, add noise
            "MoPeD (xfacta)":    0.10,
            "COOLANT (xfacta)":  0.10,
            "EMAF (xfacta)":     0.00,  # fully collapsed

            # MMHL variants: biased toward Fake regardless of input
            "MoPeD (mmhl)":      0.10,
            "COOLANT (mmhl)":    0.10,
            "EMAF (mmhl)":       0.10,

            # Weibo variants: wrong domain + vocab mismatch, essentially random
            "MoPeD (weibo)":     0.05,
            "COOLANT (weibo)":   0.05,
            "EMAF (weibo)":      0.05,
        }

        # Label-order map (option 5).
        # "RF": model returns [Real, Fake] (default)
        # "FR": model returns [Fake, Real] and needs flipping
        self.label_order_map = {label: "RF" for label in self.expert_reliability.keys()}

        # Abstain/uncertainty thresholds (option 4).
        self.min_vote_strength = 0.20
        self.min_agreement = 0.65
        
        print("✓ All 12 experts loaded and ready.")

    def _detect_language(self, text):
        """Simple language routing: detect Chinese characters, otherwise treat as English."""
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        return 'en'

    def _iter_experts(self, lang, has_real_image):
        """
        Yields (model_label, expert_instance, vote_weight) for active experts.
        Keeps soft voting, but routes away from obviously mismatched language experts
        and downweights image-heavy behavior when image is missing.
        """
        # Base reliability weights (tunable after validation)
        family_weight = {
            "MoPeD": 1.00,
            "COOLANT": 1.00,
            "EMAF": 0.95,
        }

        # Language/domain compatibility multipliers
        # Weibo experts are Chinese-oriented; others are English-oriented.
        if lang == 'zh':
            domain_weight = {"weibo": 1.00, "xfacta": 0.40, "snopes": 0.40, "mmhl": 0.40}
        else:
            domain_weight = {"weibo": 0.35, "xfacta": 1.00, "snopes": 1.00, "mmhl": 1.00}

        # No-image gating (option 1): keep voting but reduce image-dependent experts.
        if has_real_image:
            no_image_family_scale = {"MoPeD": 1.00, "COOLANT": 1.00, "EMAF": 1.00}
            no_image_domain_scale = {"weibo": 1.00, "xfacta": 1.00, "snopes": 1.00, "mmhl": 1.00}
        else:
            no_image_family_scale = {"MoPeD": 0.50, "COOLANT": 0.25, "EMAF": 0.25}
            no_image_domain_scale = {"weibo": 0.60, "xfacta": 1.00, "snopes": 1.00, "mmhl": 1.00}

        for name, expert in self.moped_experts.items():
            label = f"MoPeD ({name})"
            weight = (
                family_weight["MoPeD"] *
                domain_weight[name] *
                no_image_family_scale["MoPeD"] *
                no_image_domain_scale[name] *
                self.expert_reliability.get(label, 1.0)
            )
            if weight <= 0.0:
                continue
            yield label, expert, weight

        for name, expert in self.coolant_experts.items():
            label = f"COOLANT ({name})"
            weight = (
                family_weight["COOLANT"] *
                domain_weight[name] *
                no_image_family_scale["COOLANT"] *
                no_image_domain_scale[name] *
                self.expert_reliability.get(label, 1.0)
            )
            if weight <= 0.0:
                continue
            yield label, expert, weight

        for name, expert in self.emaf_experts.items():
            label = f"EMAF ({name})"
            weight = (
                family_weight["EMAF"] *
                domain_weight[name] *
                no_image_family_scale["EMAF"] *
                no_image_domain_scale[name] *
                self.expert_reliability.get(label, 1.0)
            )
            if weight <= 0.0:
                continue
            yield label, expert, weight

    def _has_real_image_input(self, image_path):
        """True only when caller provided an actual image (path or PIL), not fallback blank."""
        if isinstance(image_path, Image.Image):
            return True
        if isinstance(image_path, str) and image_path.strip():
            return os.path.exists(image_path)
        return False

    def _normalize_probs(self, real_p, fake_p):
        """Clamp and normalize probabilities for stable ensembling."""
        real_p = max(0.0, min(1.0, float(real_p)))
        fake_p = max(0.0, min(1.0, float(fake_p)))
        total = real_p + fake_p
        if total <= 0:
            return 0.5, 0.5
        return real_p / total, fake_p / total

    def _apply_label_order(self, model_label, real_p, fake_p):
        """
        Applies label-order correction per model (option 5).
        RF: no change; FR: swap.
        """
        order = self.label_order_map.get(model_label, "RF")
        if order == "FR":
            return fake_p, real_p
        return real_p, fake_p

    def _resolve_input_image(self, image_path=None):
        """
        Returns a PIL RGB image for inference.
        - If image_path is None, uses a blank fallback image for text-only inference.
        - If image_path is a PIL image, converts it to RGB.
        - If image_path is a filesystem path, loads and converts to RGB.
        """
        if image_path is None:
            return Image.new('RGB', (224, 224), color=(255, 255, 255))

        if isinstance(image_path, Image.Image):
            return image_path.convert('RGB')

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        return Image.open(image_path).convert('RGB')

    def get_prediction(self, text, image_path=None):
        """
        Queries active experts and performs weighted soft voting.
        Text is required. Image is optional.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` is required and must be a non-empty string.")

        has_real_image = self._has_real_image_input(image_path)
        img = self._resolve_input_image(image_path)
        lang = self._detect_language(text)
        all_results = []
        weighted_real = 0.0
        weighted_fake = 0.0
        total_weight = 0.0

        # Gather and normalize model probabilities before ensembling
        for model_label, expert, weight in self._iter_experts(lang, has_real_image):
            res = expert.predict(text, img)
            real_p, fake_p = self._normalize_probs(res['Real'], res['Fake'])
            real_p, fake_p = self._apply_label_order(model_label, real_p, fake_p)
            margin = abs(real_p - fake_p)

            row = {
                "model": model_label,
                "Real": real_p,
                "Fake": fake_p,
                "weight": float(weight),
                "margin": margin,
                "predicted_label": "Fake" if fake_p > real_p else "Real",
            }
            all_results.append(row)

            weighted_real += real_p * weight
            weighted_fake += fake_p * weight
            total_weight += weight

        if total_weight <= 0:
            raise RuntimeError("No expert weight available for voting.")

        # Weighted soft voting
        avg_real = weighted_real / total_weight
        avg_fake = weighted_fake / total_weight
        
        final_verdict = "Fake" if avg_fake > avg_real else "Real"
        vote_strength = abs(avg_fake - avg_real)  # 0..1

        # Agreement: weighted fraction of experts matching final verdict
        agreement_weight = sum(
            r["weight"] for r in all_results if r["predicted_label"] == final_verdict
        ) / total_weight

        # More trustworthy confidence than raw max(avg_real, avg_fake)
        confidence = 0.5 + 0.5 * (vote_strength * agreement_weight)

        is_uncertain = (vote_strength < self.min_vote_strength) or (agreement_weight < self.min_agreement)
        uncertainty_reason = None
        if is_uncertain:
            if vote_strength < self.min_vote_strength and agreement_weight < self.min_agreement:
                uncertainty_reason = "low_vote_strength_and_low_agreement"
            elif vote_strength < self.min_vote_strength:
                uncertainty_reason = "low_vote_strength"
            else:
                uncertainty_reason = "low_agreement"

        # "Most confident model" by margin, not raw max prob
        best_overall = max(all_results, key=lambda x: x["margin"])

        return {
            "final_verdict": "Uncertain" if is_uncertain else final_verdict,
            "overall_confidence": confidence,
            "most_confident_model": best_overall['model'],
            "best_model_score": best_overall["margin"],
            "language_detected": lang,
            "used_real_image": has_real_image,
            "vote_strength": vote_strength,
            "agreement": agreement_weight,
            "is_uncertain": is_uncertain,
            "uncertainty_reason": uncertainty_reason,
            "avg_real": avg_real,
            "avg_fake": avg_fake,
            "all_scores": all_results # Useful for debugging or detailed UI views
        }

    def benchmark_prompts(self, cases):
        """
        Run multiple prompts through the ensemble.
        Each case supports:
          - {"text": "..."}  # text-only
          - {"text": "...", "image_path": "/path/img.jpg"}  # text + image
          - optional "expected": "Real" or "Fake"
        Returns a list of per-case result dicts.
        """
        results = []
        for i, case in enumerate(cases, start=1):
            text = case.get("text", "")
            image_path = case.get("image_path")
            expected = case.get("expected")
            pred = self.get_prediction(text, image_path)

            row = {
                "case_id": i,
                "text": text,
                "expected": expected,
                "predicted": pred["final_verdict"],
                "confidence": pred["overall_confidence"],
                "language_detected": pred["language_detected"],
                "used_real_image": pred["used_real_image"],
                "vote_strength": pred["vote_strength"],
                "agreement": pred["agreement"],
                "is_uncertain": pred["is_uncertain"],
                "uncertainty_reason": pred["uncertainty_reason"],
                "most_confident_model": pred["most_confident_model"],
                "best_model_score": pred["best_model_score"],
            }
            if expected in ("Real", "Fake"):
                row["correct"] = (row["predicted"] == expected)
            results.append(row)
        return results

    def fit_label_order_from_benchmark(self, cases, min_cases=3, min_improvement=0.15, auto_apply=True):
        """
        Option 5 helper:
        Detect likely label-order inversion per expert using expected labels.
        If flipped order performs materially better than normal, mark as "FR".
        """
        labeled_cases = [c for c in cases if c.get("expected") in ("Real", "Fake")]
        if len(labeled_cases) < min_cases:
            raise ValueError(f"Need at least {min_cases} labeled cases to fit label order.")

        # Build one image per case once.
        prepared = []
        for case in labeled_cases:
            prepared.append({
                "text": case["text"],
                "expected": case["expected"],
                "img": self._resolve_input_image(case.get("image_path"))
            })

        # Collect all experts in a flat map.
        all_experts = {}
        for name, expert in self.moped_experts.items():
            all_experts[f"MoPeD ({name})"] = expert
        for name, expert in self.coolant_experts.items():
            all_experts[f"COOLANT ({name})"] = expert
        for name, expert in self.emaf_experts.items():
            all_experts[f"EMAF ({name})"] = expert

        report = {}
        for model_label, expert in all_experts.items():
            normal_correct = 0
            flipped_correct = 0
            for case in prepared:
                res = expert.predict(case["text"], case["img"])
                real_p, fake_p = self._normalize_probs(res["Real"], res["Fake"])

                normal_pred = "Fake" if fake_p > real_p else "Real"
                flipped_pred = "Fake" if real_p > fake_p else "Real"

                if normal_pred == case["expected"]:
                    normal_correct += 1
                if flipped_pred == case["expected"]:
                    flipped_correct += 1

            n = len(prepared)
            normal_acc = normal_correct / n
            flipped_acc = flipped_correct / n
            improvement = flipped_acc - normal_acc
            suggested_order = "FR" if improvement >= min_improvement else "RF"

            if auto_apply:
                self.label_order_map[model_label] = suggested_order

            report[model_label] = {
                "normal_acc": normal_acc,
                "flipped_acc": flipped_acc,
                "improvement_if_flipped": improvement,
                "suggested_order": suggested_order,
                "applied_order": self.label_order_map[model_label],
            }

        return report
    def diagnose(self, text, image_path=None, n_cases=None):
        """
        Prints raw per-expert outputs for a single input, before any
        label order correction or weighting. Use this to see if a model's
        output[0] actually corresponds to Real or Fake.
        """
        img = self._resolve_input_image(image_path)
        print(f"\nINPUT: \"{text[:70]}\"")
        print(f"{'Model':<25} {'Raw[0]':>8} {'Raw[1]':>8} {'Raw pred':>10} {'Order':>6} {'Final pred':>12}")
        print("-" * 75)
        all_experts = {}
        for name, expert in self.moped_experts.items():
            all_experts[f"MoPeD ({name})"] = expert
        for name, expert in self.coolant_experts.items():
            all_experts[f"COOLANT ({name})"] = expert
        for name, expert in self.emaf_experts.items():
            all_experts[f"EMAF ({name})"] = expert

        for label, expert in all_experts.items():
            res = expert.predict(text, img)
            raw0, raw1 = res['Real'], res['Fake']  # before any flipping
            raw_pred = "Fake" if raw1 > raw0 else "Real"
            order = self.label_order_map.get(label, "RF")
            r, f = self._apply_label_order(label, raw0, raw1)
            final_pred = "Fake" if f > r else "Real"
            print(f"  {label:<23} {raw0:>8.3f} {raw1:>8.3f} {raw_pred:>10} {order:>6} {final_pred:>12}")
manager = ModelManager()

# Run this ONCE with real labeled fake-news examples from your actual datasets
calibration_cases = [
    {"text": "COVID vaccine contains microchips", "expected": "Fake"},
    {"text": "NASA confirms moon landing in 1969", "expected": "Real"},
    {"text": "Drinking bleach cures COVID", "expected": "Fake"},
    {"text": "The Earth orbits the Sun", "expected": "Real"},
    {"text": "5G towers cause coronavirus", "expected": "Fake"},
]
report = manager.fit_label_order_from_benchmark(calibration_cases, auto_apply=True)
for model, stats in report.items():
    print(f"{model}: order={stats['applied_order']} normal={stats['normal_acc']:.0%} flipped={stats['flipped_acc']:.0%}")
# Simple test if run directly

if __name__ == "__main__":
    manager = ModelManager()

    manager.diagnose("hitler tests his new weapon of mass destruction circa colourized")
    manager.diagnose("galloping mini pony")

    print("\n--- Running Global Prediction ---")

    # Text + image
    result = manager.get_prediction(
        "Protesters storming the parliament building",
        image_path="/home/odobasia/Downloads/BERT_Project/original.jpg"
    )

    print(f"\nFINAL VERDICT: {result['final_verdict']}")
    print(f"CONFIDENCE: {result['overall_confidence']:.2%}")
    print(f"MOST CONFIDENT EXPERT: {result['most_confident_model']}")
    print(f"LANGUAGE DETECTED: {result['language_detected']}")
    print(f"USED REAL IMAGE: {result['used_real_image']}")
    print(f"VOTE STRENGTH: {result['vote_strength']:.4f}")
    print(f"AGREEMENT: {result['agreement']:.4f}")
    print(f"BEST MODEL SCORE (margin): {result['best_model_score']:.4f}")
    print(f"UNCERTAIN: {result['is_uncertain']} ({result['uncertainty_reason']})")

    # Optional quick benchmark (text-only by default; add image_path per case if desired)
    benchmark_cases = [
        {"text": "Grass cures cancer", "expected": "Fake"},
        {"text": "The earth revolves around the sun", "expected": "Real"},
        {"text": "Drinking bleach is safe for humans", "expected": "Fake"},
    ]
    benchmark_results = manager.benchmark_prompts(benchmark_cases)
    print("\n--- Benchmark (Quick) ---")
    for row in benchmark_results:
        expected_str = row["expected"] if row["expected"] else "-"
        correct_str = row.get("correct")
        if correct_str is None:
            correct_str = "-"
        print(
            f"Case {row['case_id']}: pred={row['predicted']} exp={expected_str} "
            f"conf={row['confidence']:.2%} lang={row['language_detected']} "
            f"agree={row['agreement']:.3f} uncertain={row['is_uncertain']} "
            f"model={row['most_confident_model']} correct={correct_str}"
        )
    
