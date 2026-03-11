import os
import re
import sys
from PIL import Image
import torch

sys.path.append(os.path.join(os.getcwd(), "MoPeD"))
sys.path.append(os.path.join(os.getcwd(), "MCAN"))
sys.path.append(os.path.join(os.getcwd(), "COOLANT"))
sys.path.append(os.path.join(os.getcwd(), "EMAF"))
# Import your verified wrappers
from moped_wrapper import MopedInference
from coolant_wrapper import CoolantInference
from emaf_wrapper import EmafInference
from mcan_wrapper import McanInference         


class ModelManager:
    def __init__(self):
        print("Initializing Global Model Manager...")

        self.moped_experts = {
            "xfacta": MopedInference("weights/best_moped_xfacta.pth",        dataset_type='english', device=torch.device('cuda:0')),
            "snopes": MopedInference("weights/best_moped_snopes.pth",         dataset_type='english', device=torch.device('cuda:0')),
            "weibo":  MopedInference("weights/best_moped_weibo.pth",          dataset_type='weibo',   device=torch.device('cuda:0')),
            "mmhl":   MopedInference("weights/best_moped_mmhl_fold0.pth",     dataset_type='english', device=torch.device('cuda:1')),
        }

        self.coolant_experts = {
            "xfacta": CoolantInference("weights/best_model_coolant_xfacta.pth",    device=torch.device('cuda:1')),
            "snopes": CoolantInference("weights/best_model_coolant_multimodal.pth", device=torch.device('cuda:1')),
            "weibo":  CoolantInference("weights/best_coolant_weibo.pth",            device=torch.device('cuda:2')),
            "mmhl":   CoolantInference("weights/best_model_coolant_mmhl_fold4.pth", device=torch.device('cuda:2')),
        }

        self.emaf_experts = {
            "xfacta": EmafInference("weights/best_emaf_xfacta.pth",    lang='en', device=torch.device('cuda:2')),
            "snopes": EmafInference("weights/best_emaf_multimodal.pth", lang='en', device=torch.device('cuda:3')),
            "weibo":  EmafInference("weights/best_emaf_weibo.pth",      lang='zh', device=torch.device('cuda:3')),
            "mmhl":   EmafInference("weights/best_emaf_mmhl_fold3.pth", lang='en', device=torch.device('cuda:3')),
        }

        self.mcan_experts = {
            "xfacta": McanInference("weights/best_mcan_xfacta.pth",     dataset_type='english', device=torch.device('cuda:0')),
            "snopes": McanInference("weights/best_mcan_snopes_6.pth",   dataset_type='english', device=torch.device('cuda:1')),
            # Temporarily disabled — retraining in progress (architecture divergence)
            # "weibo": McanInference("weights/best_mcan_weibo.pth",      dataset_type='weibo',   device=torch.device('cuda:2')),
            "mmhl":   McanInference("weights/best_mcan_mmhl_fold0.pth", dataset_type='english', device=torch.device('cuda:3')),
        }
        # Tuned from validation benchmarks.
        self.expert_reliability = {
            # Snopes: High Signal
            "MoPeD (snopes)"   : 1.00,
            "COOLANT (snopes)" : 1.00,
            "MCAN (snopes)"    : 1.00,
            "EMAF (snopes)"    : 0.20,

            # XFacta. Some collapsed toward Real
            "MoPeD (xfacta)"   : 0.10,
            "COOLANT (xfacta)" : 0.10,
            "MCAN (xfacta)"    : 0.00, # fully collapsed
            "EMAF (xfacta)"    : 0.00, # fully collapsed

            # MMHL. 1 Biased toward Fake 
            "MoPeD (mmhl)"     : 0.10,
            "COOLANT (mmhl)"   : 0.10,
            "EMAF (mmhl)"      : 0.10,
            "MCAN (mmhl)"      : 0.00,

            #Weibo. This one had Mixed / Weak Signal ─
            #  "weibo" temporarily disabled — retraining in progress
            # "MCAN (weibo)"     : 0.60,
            "EMAF (weibo)"     : 0.10, # weak but correct direction
            "COOLANT (weibo)"  : 0.05, # near coin-flip
            "MoPeD (weibo)"    : 0.00, # zero signal (frozen)
        }

        # ── Label-order map ───────────────────────────────────────────────────
        # "RF": model returns [Real, Fake] (default)
        # "FR": model returns [Fake, Real] and needs flipping
        self.label_order_map = {label: "RF" for label in self.expert_reliability.keys()}

        # ── Abstain / uncertainty thresholds ─────────────────────────────────
        self.min_vote_strength = 0.20
        self.min_agreement     = 0.65

        print("All experts loaded and ready.")  

    # Language routing since some experts are trained on Chinese data and may underperform on English, and vice versa.
    def _detect_language(self, text):
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        return 'en'

    # Expert iterator which will apply dynamic weighting based on language and image presence.
    def _iter_experts(self, lang):
        family_weight = {
            "MoPeD":   1.00,
            "COOLANT": 1.00,
            "EMAF":    0.95,
            "MCAN":    1.00,  
        }

        # Check if language-specific domain weights are defined, otherwise default to 1.0
        if lang == 'zh':
            domain_weight = {"weibo": 1.00, "xfacta": 0.40, "snopes": 0.40, "mmhl": 0.40}
        else:
            domain_weight = {"weibo": 0.35, "xfacta": 1.00, "snopes": 1.00, "mmhl": 1.00}
        # For each model, calculate the final weight as: family_weight * domain_weight * expert_reliability
        for name, expert in self.moped_experts.items():
            label  = f"MoPeD ({name})"
            weight = (
                family_weight["MoPeD"] *
                domain_weight[name] *
                self.expert_reliability.get(label, 1.0)
            )
            if weight > 0.0:
                yield label, expert, weight

        for name, expert in self.coolant_experts.items():
            label  = f"COOLANT ({name})"
            weight = (
                family_weight["COOLANT"] *
                domain_weight[name] *
                self.expert_reliability.get(label, 1.0)
            )
            if weight > 0.0:
                yield label, expert, weight

        for name, expert in self.emaf_experts.items():
            label  = f"EMAF ({name})"
            weight = (
                family_weight["EMAF"] *
                domain_weight[name] *
                self.expert_reliability.get(label, 1.0)
            )
            if weight > 0.0:
                yield label, expert, weight

        for name, expert in self.mcan_experts.items():
            label  = f"MCAN ({name})"
            weight = (
                family_weight["MCAN"] *
                domain_weight[name] *
                self.expert_reliability.get(label, 1.0)
            )
            if weight > 0.0:
                yield label, expert, weight

    # Hanldes the input image to make sure we always have a valid PIL RGB image to pass to the experts, even if the user doesn't provide one or provides an invalid path. 
    # This allows the ensemble to still function (with lower confidence) in text-only scenarios
    def _resolve_input_image(self, image_path=None):
        """
        Returns a PIL RGB image for inference.
        - If image_path is None, uses a blank fallback for text-only inference.
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

    # Probability helpers. These will clamp to [0,1] and normalize to sum to 1, which is important for stable ensembling and to prevent any single expert from dominating due to scale issues.
    def _normalize_probs(self, real_p, fake_p):
        """Clamp and normalize probabilities for stable ensembling."""
        real_p = max(0.0, min(1.0, float(real_p)))
        fake_p = max(0.0, min(1.0, float(fake_p)))
        total  = real_p + fake_p
        if total <= 0:
            return 0.5, 0.5
        return real_p / total, fake_p / total

    # RF: no change; FR: swap
    def _apply_label_order(self, model_label, real_p, fake_p):
        order = self.label_order_map.get(model_label, "RF")
        if order == "FR":
            return fake_p, real_p
        return real_p, fake_p

    # This is the Core prediction. It will: 
    # 1) validate inputs, 2) detect language, 3) iterate experts with dynamic weighting, 
    # 4) aggregate votes, 5) calculate confidence and uncertainty, and 6) return a detailed result dict for downstream use and analysis.
    def get_prediction(self, text, image_path=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` is required and must be a non-empty string.")
        if image_path is None:
            raise ValueError("`image_path` is required — this ensemble expects both text and image.")
        img  = self._resolve_input_image(image_path)
        lang = self._detect_language(text)

        all_results   = []
        weighted_real = 0.0
        weighted_fake = 0.0
        total_weight  = 0.0

        for model_label, expert, weight in self._iter_experts(lang):
            res             = expert.predict(text, img)
            real_p, fake_p  = self._normalize_probs(res['Real'], res['Fake'])
            real_p, fake_p  = self._apply_label_order(model_label, real_p, fake_p)
            margin          = abs(real_p - fake_p)

            row = {
                "model":           model_label,
                "Real":            real_p,
                "Fake":            fake_p,
                "weight":          float(weight),
                "margin":          margin,
                "predicted_label": "Fake" if fake_p > real_p else "Real",
            }
            all_results.append(row)
            weighted_real += real_p * weight
            weighted_fake += fake_p * weight
            total_weight  += weight

        if total_weight <= 0:
            raise RuntimeError("No expert weight available for voting.")

        avg_real     = weighted_real / total_weight
        avg_fake     = weighted_fake / total_weight
        final_verdict = "Fake" if avg_fake > avg_real else "Real"
        vote_strength = abs(avg_fake - avg_real)

        agreement_weight = sum(
            r["weight"] for r in all_results if r["predicted_label"] == final_verdict
        ) / total_weight

        confidence  = 0.5 + 0.5 * (vote_strength * agreement_weight)
        is_uncertain = (vote_strength < self.min_vote_strength) or (agreement_weight < self.min_agreement)

        uncertainty_reason = None
        if is_uncertain:
            if vote_strength < self.min_vote_strength and agreement_weight < self.min_agreement:
                uncertainty_reason = "low_vote_strength_and_low_agreement"
            elif vote_strength < self.min_vote_strength:
                uncertainty_reason = "low_vote_strength"
            else:
                uncertainty_reason = "low_agreement"

        best_overall = max(all_results, key=lambda x: x["margin"])

        return {
            "final_verdict":       "Uncertain" if is_uncertain else final_verdict,
            "overall_confidence":  confidence,
            "most_confident_model": best_overall['model'],
            "best_model_score":    best_overall["margin"],
            "language_detected":   lang,
            "used_real_image":     True,  
            "vote_strength":       vote_strength,
            "agreement":           agreement_weight,
            "is_uncertain":        is_uncertain,
            "uncertainty_reason":  uncertainty_reason,
            "avg_real":            avg_real,
            "avg_fake":            avg_fake,
            "all_scores":          all_results,
        }

    # This is the Batch benchmark for when we want to run multiple cases through the ensemble and get a structured report. 
    # Each case can only be text+image, and can optionally include an expected label for accuracy calculation. 
    # The output is a list of dicts with detailed results for each case, which can be used for analysis and further tuning.
    def benchmark_prompts(self, cases):
        results = []
        for i, case in enumerate(cases, start=1):
            text       = case.get("text", "")
            image_path = case.get("image_path")
            expected   = case.get("expected")
            pred       = self.get_prediction(text, image_path)

            row = {
                "case_id":            i,
                "text":               text,
                "expected":           expected,
                "predicted":          pred["final_verdict"],
                "confidence":         pred["overall_confidence"],
                "language_detected":  pred["language_detected"],
                "vote_strength":      pred["vote_strength"],
                "agreement":          pred["agreement"],
                "is_uncertain":       pred["is_uncertain"],
                "uncertainty_reason": pred["uncertainty_reason"],
                "most_confident_model": pred["most_confident_model"],
                "best_model_score":   pred["best_model_score"],
            }
            if expected in ("Real", "Fake"):
                row["correct"] = (row["predicted"] == expected)
            results.append(row)
        return results

    # This function will analyze a set of labeled cases to detect if any experts are likely returning flipped probabilities (Fake, Real instead of Real, Fake).
    def fit_label_order_from_benchmark(self, cases, min_cases=3, min_improvement=0.15, auto_apply=True):
        labeled_cases = [c for c in cases if c.get("expected") in ("Real", "Fake")]
        if len(labeled_cases) < min_cases:
            raise ValueError(f"Need at least {min_cases} labeled cases to fit label order.")

        prepared = []
        for case in labeled_cases:
            prepared.append({
                "text":     case["text"],
                "expected": case["expected"],
                "img":      self._resolve_input_image(case.get("image_path")),
            })

        all_experts = self._all_experts_flat()

        report = {}
        for model_label, expert in all_experts.items():
            normal_correct  = 0
            flipped_correct = 0
            for case in prepared:
                res             = expert.predict(case["text"], case["img"])
                real_p, fake_p  = self._normalize_probs(res["Real"], res["Fake"])
                normal_pred     = "Fake" if fake_p > real_p else "Real"
                flipped_pred    = "Fake" if real_p > fake_p else "Real"
                if normal_pred  == case["expected"]: normal_correct  += 1
                if flipped_pred == case["expected"]: flipped_correct += 1

            n = len(prepared)
            normal_acc = normal_correct  / n
            flipped_acc = flipped_correct / n
            improvement  = flipped_acc - normal_acc
            suggested_order = "FR" if improvement >= min_improvement else "RF"

            if auto_apply:
                self.label_order_map[model_label] = suggested_order

            report[model_label] = {
                "normal_acc":          normal_acc,
                "flipped_acc":         flipped_acc,
                "improvement_if_flipped": improvement,
                "suggested_order":     suggested_order,
                "applied_order":       self.label_order_map[model_label],
            }
        return report

    # Diagnose will print raw per-expert outputs for a single input, before any label-order correction or weighting.
    def diagnose(self, text, image_path=None):
        img = self._resolve_input_image(image_path)
        print(f"\nINPUT: \"{text[:70]}\"")
        print(f"{'Model':<25} {'Raw[0]':>8} {'Raw[1]':>8} {'Raw pred':>10} {'Order':>6} {'Final pred':>12}")
        print("-" * 75)

        for label, expert in self._all_experts_flat().items():
            res        = expert.predict(text, img)
            raw0, raw1 = res['Real'], res['Fake']
            raw_pred   = "Fake" if raw1 > raw0 else "Real"
            order      = self.label_order_map.get(label, "RF")
            r, f       = self._apply_label_order(label, raw0, raw1)
            final_pred = "Fake" if f > r else "Real"
            print(f"  {label:<23} {raw0:>8.3f} {raw1:>8.3f} {raw_pred:>10} {order:>6} {final_pred:>12}")

    # ── Internal helper: flat expert dict ────────────────────────────────────
    def _all_experts_flat(self):
        """Returns a single ordered dict of all experts across all families."""
        experts = {}
        for name, expert in self.moped_experts.items():
            experts[f"MoPeD ({name})"]   = expert
        for name, expert in self.coolant_experts.items():
            experts[f"COOLANT ({name})"] = expert
        for name, expert in self.emaf_experts.items():
            experts[f"EMAF ({name})"]    = expert
        for name, expert in self.mcan_experts.items():      # ← NEW
            experts[f"MCAN ({name})"]    = expert
        return experts

manager = ModelManager()


# ── Quick smoke-test when run directly ────────────────────────────────────────
if __name__ == "__main__":
    manager = ModelManager()

    manager.diagnose("hitler tests his new weapon of mass destruction circa colourized")
    manager.diagnose("galloping mini pony")

    print("\n--- Running Global Prediction ---")
    result = manager.get_prediction(
        "Protesters storming the parliament building",
        image_path="/home/odobasia/Downloads/BERT_Project/original.jpg",
    )
    print(f"\nFINAL VERDICT:          {result['final_verdict']}")
    print(f"CONFIDENCE:             {result['overall_confidence']:.2%}")
    print(f"MOST CONFIDENT EXPERT:  {result['most_confident_model']}")
    print(f"LANGUAGE DETECTED:      {result['language_detected']}")
    print(f"VOTE STRENGTH:          {result['vote_strength']:.4f}")
    print(f"AGREEMENT:              {result['agreement']:.4f}")
    print(f"BEST MODEL SCORE:       {result['best_model_score']:.4f}")
    print(f"UNCERTAIN:              {result['is_uncertain']} ({result['uncertainty_reason']})")

    benchmark_cases = [
        {"text": "Grass cures cancer",                          "expected": "Fake"},
        {"text": "The earth revolves around the sun",           "expected": "Real"},
        {"text": "Drinking bleach is safe for humans",          "expected": "Fake"},
    ]
    benchmark_results = manager.benchmark_prompts(benchmark_cases)
    print("\n--- Benchmark (Quick) ---")
    for row in benchmark_results:
        expected_str = row["expected"] if row["expected"] else "-"
        correct_str  = row.get("correct", "-")
        print(
            f"Case {row['case_id']}: pred={row['predicted']} exp={expected_str} "
            f"conf={row['confidence']:.2%} lang={row['language_detected']} "
            f"agree={row['agreement']:.3f} uncertain={row['is_uncertain']} "
            f"model={row['most_confident_model']} correct={correct_str}"
        )
