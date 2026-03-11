import os
import re
import sys
from PIL import Image
import torch

# Path setup 
sys.path.append(os.path.join(os.getcwd(), "MoPeD"))
sys.path.append(os.path.join(os.getcwd(), "COOLANT"))
sys.path.append(os.path.join(os.getcwd(), "EMAF"))

#  Wrapper imports
from moped_wrapper   import MopedInference
from coolant_wrapper import CoolantInference
from emaf_wrapper    import EmafInference
from mcan_wrapper    import McanInference

from spotfake_wrapper_xfacta import SpotFakeInferenceXFacta
from spotfake_wrapper_weibo  import SpotFakeInferenceWeibo
from spotfake_wrapper        import SpotFakeInference        

from mvae_wrapper_xfacta import MVAEInferenceXFacta
from mvae_wrapper_weibo  import MVAEInferenceWeibo
from mvae_wrapper        import MVAEInference                

from attrnn_wrapper_xfacta import AttRNNInferenceXFacta
from attrnn_wrapper_weibo  import AttRNNInferenceWeibo
from attrnn_wrapper        import AttRNNInference                  


class ModelManager:
    def __init__(self):
        print("Initializing Global Model Manager...")

        # MoPeD 
        self.moped_experts = {
            "xfacta": MopedInference("weights/best_moped_xfacta.pth",    dataset_type='english', device=torch.device('cuda:0')),
            "snopes": MopedInference("weights/best_moped_snopes.pth",     dataset_type='english', device=torch.device('cuda:0')),
            "weibo":  MopedInference("weights/best_moped_weibo.pth",      dataset_type='weibo',   device=torch.device('cuda:0')),
            "mmhl":   MopedInference("weights/best_moped_mmhl_fold0.pth", dataset_type='english', device=torch.device('cuda:1')),
        }

        # COOLANT 
        self.coolant_experts = {
            "xfacta": CoolantInference("weights/best_model_coolant_xfacta.pth",    device=torch.device('cuda:0')),
            "snopes": CoolantInference("weights/best_model_coolant_multimodal.pth", device=torch.device('cuda:1')),
            "weibo":  CoolantInference("weights/best_coolant_weibo.pth",            device=torch.device('cuda:2')),
            "mmhl":   CoolantInference("weights/best_model_coolant_mmhl_fold4.pth", device=torch.device('cuda:2')),
        }

        # EMAF
        self.emaf_experts = {
            "xfacta": EmafInference("weights/best_emaf_xfacta.pth",    lang='en', device=torch.device('cuda:2')),
            "snopes": EmafInference("weights/best_emaf_multimodal.pth", lang='en', device=torch.device('cuda:3')),
            "weibo":  EmafInference("weights/best_emaf_weibo.pth",      lang='zh', device=torch.device('cuda:3')),
            "mmhl":   EmafInference("weights/best_emaf_mmhl_fold3.pth", lang='en', device=torch.device('cuda:3')),
        }

        # MCAN 
        # dataset_type controls BERT variant: 'weibo' → chinese BERT, else bert-base-uncased
        self.mcan_experts = {
            "xfacta": McanInference("weights/best_mcan_xfacta.pth",     dataset_type='english', device=torch.device('cuda:0')),
            "snopes": McanInference("weights/best_mcan_snopes_6.pth",   dataset_type='english', device=torch.device('cuda:3')),
            "weibo": McanInference("weights/best_mcan_weibo.pth",      dataset_type='weibo',   device=torch.device('cuda:2')),
            "mmhl":   McanInference("weights/best_mcan_mmhl_fold0.pth", dataset_type='english', device=torch.device('cuda:3')),
        }

        # SpotFake 
        # weibo: collapsed Fake (0.794 Fake on real input) → reliability 0.00
        # mmhl:  collapsed Real (0.978 Real on fake input) → reliability 0.00
        self.spotfake_experts = {
            "xfacta": SpotFakeInferenceXFacta("weights/spotfake_xfacta.pth", device=torch.device('cuda:0')),
            "weibo":  SpotFakeInferenceWeibo("weights/spotfake_weibo.pth",   device=torch.device('cuda:1')),
            "mmhl":   SpotFakeInference("weights/best_spotfake_med.pth",     device=torch.device('cuda:2')),
        }

        # MVAE
        # All three variants discriminate correctly — full weight pending benchmark
        self.mvae_experts = {
            "xfacta": MVAEInferenceXFacta("weights/mvae_xfacta.pth",  device=torch.device('cuda:2')),
            "weibo":  MVAEInferenceWeibo("weights/mvae_weibo.pth",     device=torch.device('cuda:3')),
            "mmhl":   MVAEInference("weights/best_mvae_med.pth",       device=torch.device('cuda:3')),
        }

        # ATTRNN
        # weibo: collapsed Fake (0.918 Fake on real input) → reliability 0.00
        # mmhl:  weak discrimination both ways             → reliability 0.00
        self.attrnn_experts = {
            "xfacta": AttRNNInferenceXFacta("weights/attrnn_xfacta.pth", device=torch.device('cuda:0')),
            "weibo":  AttRNNInferenceWeibo("weights/attrnn_weibo.pth",   device=torch.device('cuda:1')),
            "mmhl":   AttRNNInference("weights/best_attrnn_med.pth",     device=torch.device('cuda:2')),
        }

        # Per-expert reliability weights
        self.expert_reliability = {

            # MoPeD
            "MoPeD (snopes)":    1.00,  # primary anchor
            "MoPeD (xfacta)":    0.10,  # collapsed toward Real
            "MoPeD (weibo)":     0.00,  # frozen at 0.647/0.353 on every input
            "MoPeD (mmhl)":      0.10,  # biased toward Fake

            # COOLANT 
            "COOLANT (snopes)":  1.00,  # primary anchor
            "COOLANT (xfacta)":  0.10,  # collapsed toward Real
            "COOLANT (weibo)":   0.05,  # near coin-flip
            "COOLANT (mmhl)":    0.10,  # biased toward Fake

            # EMAF 
            "EMAF (snopes)":     0.20,  # collapsed Real but soft counterweight
            "EMAF (xfacta)":     0.00,  # fully collapsed
            "EMAF (weibo)":      0.10,  # weak but correct direction
            "EMAF (mmhl)":       0.10,  # biased toward Fake

            # MCAN
            "MCAN (snopes)":     1.00,  # best discriminator (gap=0.985)
            "MCAN (xfacta)":     0.00,  # collapsed Fake
            # "MCAN (weibo)":    0.60,  # disabled pending retrain
            "MCAN (mmhl)":       0.00,  # collapsed Fake

            # SpotFake 
            "SpotFake (xfacta)": 1.00,  # perfect discrimination on anchor cases
            "SpotFake (weibo)":  0.00,  # collapsed Fake
            "SpotFake (mmhl)":   0.00,  # collapsed Real

            # MVAE
            "MVAE (xfacta)":     1.00,  # strong discrimination
            "MVAE (weibo)":      0.00,  # collapsed R=0.00 F=1.00 on all inputs
            "MVAE (mmhl)":       0.00,  # collapsed R=0.00 F=1.00 on all inputs

            # ATTRNN 
            "ATTRNN (xfacta)":   1.00,  # good discrimination
            "ATTRNN (weibo)":    0.00,  # collapsed Fake
            "ATTRNN (mmhl)":     0.00,  # weak discrimination both ways
        }

        # RF: model returns [Real, Fake] — confirmed for all via diagnose()
        self.label_order_map = {label: "RF" for label in self.expert_reliability.keys()}

        # Abstain / uncertainty thresholds 
        self.min_vote_strength = 0.20
        self.min_agreement     = 0.65

        print("✓ All experts loaded and ready.")

    # Language routing
    def _detect_language(self, text):
        """Detect Chinese characters → 'zh', otherwise → 'en'."""
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        return 'en'

    def _iter_experts(self, lang, has_real_image):
        family_weight = {
            "MoPeD":    1.00,
            "COOLANT":  1.00,
            "EMAF":     0.95,
            "MCAN":     1.00,
            "SpotFake": 1.00,
            "MVAE":     1.00,
            "ATTRNN":   1.00,
        }

        if lang == 'zh':
            domain_weight = {"weibo": 1.00, "xfacta": 0.40, "snopes": 0.40, "mmhl": 0.40}
        else:
            domain_weight = {"weibo": 0.35, "xfacta": 1.00, "snopes": 1.00, "mmhl": 1.00}

        all_families = [
            ("MoPeD",    self.moped_experts),
            ("COOLANT",  self.coolant_experts),
            ("EMAF",     self.emaf_experts),
            ("MCAN",     self.mcan_experts),
            ("SpotFake", self.spotfake_experts),
            ("MVAE",     self.mvae_experts),
            ("ATTRNN",   self.attrnn_experts),
        ]

        for family, experts in all_families:
            for name, expert in experts.items():
                label  = f"{family} ({name})"
                weight = (
                    family_weight[family] *
                    domain_weight.get(name, 1.0) *
                    self.expert_reliability.get(label, 1.0)
                )
                if weight > 0.0:
                    yield label, expert, weight

    #  Image helpers
    def _has_real_image_input(self, image_path):
        """True only when caller provided an actual image (path or PIL)."""
        if isinstance(image_path, Image.Image):
            return True
        if isinstance(image_path, str) and image_path.strip():
            return os.path.exists(image_path)
        return False

    def _resolve_input_image(self, image_path=None):
        """
        Returns a PIL RGB image for inference.
        Accepts None (blank fallback), PIL image, or filesystem path.
        """
        if image_path is None:
            return Image.new('RGB', (224, 224), color=(255, 255, 255))
        if isinstance(image_path, Image.Image):
            return image_path.convert('RGB')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return Image.open(image_path).convert('RGB')

    # Probability helpers 
    def _normalize_probs(self, real_p, fake_p):
        # Clamp and normalize probabilities for stable ensembling
        real_p = max(0.0, min(1.0, float(real_p)))
        fake_p = max(0.0, min(1.0, float(fake_p)))
        total  = real_p + fake_p
        if total <= 0:
            return 0.5, 0.5
        return real_p / total, fake_p / total

    def _apply_label_order(self, model_label, real_p, fake_p):
        """RF: no change; FR: swap."""
        if self.label_order_map.get(model_label, "RF") == "FR":
            return fake_p, real_p
        return real_p, fake_p

    # Core prediction
    def get_prediction(self, text, image_path=None):
        # Queries active experts and performs weighted soft voting. Both text and image are required.
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` is required and must be a non-empty string.")
        if not self._has_real_image_input(image_path):
            raise ValueError("`image_path` is required — this ensemble expects both text and image.")

        img  = self._resolve_input_image(image_path)
        lang = self._detect_language(text)

        all_results   = []
        weighted_real = 0.0
        weighted_fake = 0.0
        total_weight  = 0.0

        for model_label, expert, weight in self._iter_experts(lang, True):
            res             = expert.predict(text, img)
            real_p, fake_p  = self._normalize_probs(res['Real'], res['Fake'])
            real_p, fake_p  = self._apply_label_order(model_label, real_p, fake_p)
            margin          = abs(real_p - fake_p)

            all_results.append({
                "model":           model_label,
                "Real":            real_p,
                "Fake":            fake_p,
                "weight":          float(weight),
                "margin":          margin,
                "predicted_label": "Fake" if fake_p > real_p else "Real",
            })
            weighted_real += real_p * weight
            weighted_fake += fake_p * weight
            total_weight  += weight

        if total_weight <= 0:
            raise RuntimeError("No expert weight available for voting.")

        avg_real      = weighted_real / total_weight
        avg_fake      = weighted_fake / total_weight
        final_verdict = "Fake" if avg_fake > avg_real else "Real"
        vote_strength = abs(avg_fake - avg_real)

        agreement_weight = sum(
            r["weight"] for r in all_results if r["predicted_label"] == final_verdict
        ) / total_weight

        confidence   = 0.5 + 0.5 * (vote_strength * agreement_weight)
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
            "final_verdict":        "Uncertain" if is_uncertain else final_verdict,
            "overall_confidence":   confidence,
            "most_confident_model": best_overall['model'],
            "best_model_score":     best_overall["margin"],
            "language_detected":    lang,
            "used_real_image":      True,
            "vote_strength":        vote_strength,
            "agreement":            agreement_weight,
            "is_uncertain":         is_uncertain,
            "uncertainty_reason":   uncertainty_reason,
            "avg_real":             avg_real,
            "avg_fake":             avg_fake,
            "all_scores":           all_results,
        }

    # Batch benchmark
    # It will be important to run `fit_label_order_from_benchmark` on any new batch of cases before interpreting these results, 
    # to make sure that label-order corrections are properly applied
    def benchmark_prompts(self, cases):
        results = []
        for i, case in enumerate(cases, start=1):
            text       = case.get("text", "")
            image_path = case.get("image_path")
            expected   = case.get("expected")
            pred       = self.get_prediction(text, image_path)

            row = {
                "case_id":              i,
                "text":                 text,
                "expected":             expected,
                "predicted":            pred["final_verdict"],
                "confidence":           pred["overall_confidence"],
                "language_detected":    pred["language_detected"],
                "used_real_image":      pred["used_real_image"],
                "vote_strength":        pred["vote_strength"],
                "agreement":            pred["agreement"],
                "is_uncertain":         pred["is_uncertain"],
                "uncertainty_reason":   pred["uncertainty_reason"],
                "most_confident_model": pred["most_confident_model"],
                "best_model_score":     pred["best_model_score"],
            }
            if expected in ("Real", "Fake"):
                row["correct"] = (row["predicted"] == expected)
            results.append(row)
        return results

    # Label-order auto-fit
    def fit_label_order_from_benchmark(self, cases, min_cases=3, min_improvement=0.15, auto_apply=True):
        # Detect likely label-order inversion per expert using expected labels.
        # Must be called MANUALLY with real image+text cases — never at module load time.
        
        labeled_cases = [c for c in cases if c.get("expected") in ("Real", "Fake")]
        if len(labeled_cases) < min_cases:
            raise ValueError(f"Need at least {min_cases} labeled cases to fit label order.")

        prepared = [{
            "text":     c["text"],
            "expected": c["expected"],
            "img":      self._resolve_input_image(c.get("image_path")),
        } for c in labeled_cases]

        report = {}
        for model_label, expert in self._all_experts_flat().items():
            normal_correct  = 0
            flipped_correct = 0
            for case in prepared:
                res            = expert.predict(case["text"], case["img"])
                real_p, fake_p = self._normalize_probs(res["Real"], res["Fake"])
                if ("Fake" if fake_p > real_p else "Real") == case["expected"]: normal_correct  += 1
                if ("Fake" if real_p > fake_p else "Real") == case["expected"]: flipped_correct += 1

            n               = len(prepared)
            normal_acc      = normal_correct  / n
            flipped_acc     = flipped_correct / n
            improvement     = flipped_acc - normal_acc
            suggested_order = "FR" if improvement >= min_improvement else "RF"

            if auto_apply:
                self.label_order_map[model_label] = suggested_order

            report[model_label] = {
                "normal_acc":             normal_acc,
                "flipped_acc":            flipped_acc,
                "improvement_if_flipped": improvement,
                "suggested_order":        suggested_order,
                "applied_order":          self.label_order_map[model_label],
            }
        return report

    # Diagnose
    # This will print detailed per-expert outputs for a single input, before any label-order correction or weighting, to help with error analysis and sanity checks. 
    # It has to be called with real image+text inputs to be meaningful.
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

    # Internal helper: flat expert dict
    def _all_experts_flat(self):
        experts = {}
        for family, family_dict in [
            ("MoPeD",    self.moped_experts),
            ("COOLANT",  self.coolant_experts),
            ("EMAF",     self.emaf_experts),
            ("MCAN",     self.mcan_experts),
            ("SpotFake", self.spotfake_experts),
            ("MVAE",     self.mvae_experts),
            ("ATTRNN",   self.attrnn_experts),
        ]:
            for name, expert in family_dict.items():
                experts[f"{family} ({name})"] = expert
        return experts

if __name__ == "__main__":
    manager = ModelManager()
    manager.diagnose("hitler tests his new weapon...")
    manager.diagnose("galloping mini pony")
