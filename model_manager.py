import os
import re
import sys
from collections import OrderedDict
from PIL import Image
import torch

# Path setup 
# sys.path.append(os.path.join(os.getcwd(), "MoPeD"))    # disabled for CPU demo
# sys.path.append(os.path.join(os.getcwd(), "COOLANT"))  # disabled for CPU demo
# sys.path.append(os.path.join(os.getcwd(), "EMAF"))     # disabled for CPU demo

# #  Wrapper imports
# from moped_wrapper   import MopedInference    # disabled for CPU demo
# from coolant_wrapper import CoolantInference  # disabled for CPU demo
# from emaf_wrapper    import EmafInference     # disabled for CPU demo
# from mcan_wrapper    import McanInference     # disabled for CPU demo

from spotfake_wrapper_xfacta import SpotFakeInferenceXFacta
from spotfake_wrapper_weibo  import SpotFakeInferenceWeibo
from spotfake_wrapper        import SpotFakeInference        

from mvae_wrapper_xfacta import MVAEInferenceXFacta
from mvae_wrapper_weibo  import MVAEInferenceWeibo
from mvae_wrapper        import MVAEInference                

from attrnn_wrapper_xfacta import AttRNNInferenceXFacta
from attrnn_wrapper_weibo  import AttRNNInferenceWeibo
from attrnn_wrapper        import AttRNNInference                  

from spotfake_wrapper_snopes import SpotFakeInferenceSnopes
from mvae_wrapper_snopes     import MVAEInferenceSnopes
from attrnn_wrapper_snopes   import AttRNNInferenceSnopes

class ModelManager:
    def _download_weights(self):
        from huggingface_hub import hf_hub_download
        import os

        repo_id = "amerodobasic1/MOSIAC_Demo_Models"
        token = os.environ.get("HF_TOKEN")
        files = [
            "attrnn_xfacta.pth",
            "mvae_xfacta.pth",
            "spotfake_xfacta.pth",
        ]

        os.makedirs("weights", exist_ok=True)

        for filename in files:
            dest = os.path.join("weights", filename)
            if not os.path.exists(dest):
                print(f"Downloading {filename} from Hugging Face...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir="weights",
                    token=token,
                )
                print(f"{filename} gtg")
            else:
                print(f"{filename} already present, skipping download")

    def __init__(self):
        print("Initializing Global Model Manager...")
        # CPU demo mode: 3 experts only. See full system for 28-expert multi-GPU setup.

        # For the demo, the weights from huggingface needs to be downloaded if not already present 
        self._download_weights()

        # MoPeD 
        self.moped_experts = {
            # "xfacta": MopedInference("weights/best_moped_xfacta.pth",    dataset_type='english', device=torch.device('cpu')),
            # "snopes": MopedInference("weights/best_moped_snopes.pth",     dataset_type='english', device=torch.device('cpu')),
            # "weibo":  MopedInference("weights/best_moped_weibo.pth",      dataset_type='weibo',   device=torch.device('cpu')),
            # "mmhl":   MopedInference("weights/best_moped_mmhl_fold0.pth", dataset_type='english', device=torch.device('cpu')),
        }

        # COOLANT 
        self.coolant_experts = {
            # "xfacta": CoolantInference("weights/best_model_coolant_xfacta.pth",    device=torch.device('cpu')),
            # "snopes": CoolantInference("weights/best_model_coolant_multimodal.pth", device=torch.device('cpu')),
            # "weibo":  CoolantInference("weights/best_coolant_weibo.pth",            device=torch.device('cpu')),
            # "mmhl":   CoolantInference("weights/best_model_coolant_mmhl_fold4.pth", device=torch.device('cpu')),
        }

        # EMAF
        self.emaf_experts = {
            # "xfacta": EmafInference("weights/best_emaf_xfacta.pth",    lang='en', device=torch.device('cpu')),
            # "snopes": EmafInference("weights/best_emaf_multimodal.pth", lang='en', device=torch.device('cpu')),
            # "weibo":  EmafInference("weights/best_emaf_weibo.pth",      lang='zh', device=torch.device('cpu')),
            # "mmhl":   EmafInference("weights/best_emaf_mmhl_fold3.pth", lang='en', device=torch.device('cpu')),
        }

        # MCAN
        self.mcan_experts = {
            # "xfacta": McanInference("weights/best_mcan_xfacta.pth",    dataset_type='english', device=torch.device('cpu')),
            # "snopes": McanInference("weights/best_mcan_snopes_6.pth",   dataset_type='english', device=torch.device('cpu')),
            # "weibo":  McanInference("weights/best_mcan_weibo.pth",      dataset_type='weibo',   device=torch.device('cpu')),
            # "mmhl":   McanInference("weights/best_mcan_mmhl_fold0.pth", dataset_type='english', device=torch.device('cpu')),
        }

        # SpotFake 
        self.spotfake_experts = {
            "xfacta": SpotFakeInferenceXFacta("weights/spotfake_xfacta.pth", device=torch.device('cpu')),
            # "weibo":  SpotFakeInferenceWeibo("weights/spotfake_weibo.pth",   device=torch.device('cpu')),
            # "mmhl":   SpotFakeInference("weights/best_spotfake_med.pth",     device=torch.device('cpu')),
            # "snopes": SpotFakeInferenceSnopes("weights/spotfake_snopes.pth", device=torch.device('cpu')),
        }

        # MVAE
        self.mvae_experts = {
            "xfacta": MVAEInferenceXFacta("weights/mvae_xfacta.pth",  device=torch.device('cpu')),
            # "weibo":  MVAEInferenceWeibo("weights/mvae_weibo.pth",     device=torch.device('cpu')),
            # "mmhl":   MVAEInference("weights/best_mvae_med.pth",       device=torch.device('cpu')),
            # "snopes": MVAEInferenceSnopes("weights/mvae_snopes.pth",   device=torch.device('cpu')),
        }

        # ATTRNN 
        self.attrnn_experts = {
            "xfacta": AttRNNInferenceXFacta("weights/attrnn_xfacta.pth", device=torch.device('cpu')),
            # "weibo":  AttRNNInferenceWeibo("weights/attrnn_weibo.pth",   device=torch.device('cpu')),
            # "mmhl":   AttRNNInference("weights/best_attrnn_med.pth",     device=torch.device('cpu')),
            # "snopes": AttRNNInferenceSnopes("weights/attrnn_snopes.pth", device=torch.device('cpu')),
        }

        # Per-expert reliability weights
        self.expert_reliability = {
            # MoPeD
            "MoPeD (snopes)":     0.53,
            "MoPeD (xfacta)":     0.33,
            "MoPeD (weibo)":      0.13,
            "MoPeD (mmhl)":       0.27,
            # COOLANT
            "COOLANT (snopes)":   0.60,
            "COOLANT (xfacta)":   0.40,
            "COOLANT (weibo)":    0.00,
            "COOLANT (mmhl)":     0.00,
            # EMAF
            "EMAF (snopes)":      0.33,
            "EMAF (xfacta)":      0.33,
            "EMAF (weibo)":       0.20,
            "EMAF (mmhl)":        0.00,
            # MCAN
            "MCAN (snopes)":      0.67,
            "MCAN (xfacta)":      0.20,
            "MCAN (weibo)":       0.20,
            "MCAN (mmhl)":        0.00,
            # SpotFake
            "SpotFake (xfacta)":  0.33,
            "SpotFake (weibo)":   0.13,
            "SpotFake (mmhl)":    0.07,
            "SpotFake (snopes)":  0.00,
            # MVAE
            "MVAE (xfacta)":      0.60,
            "MVAE (weibo)":       0.13,
            "MVAE (mmhl)":        0.00,
            "MVAE (snopes)":      0.00,
            # ATTRNN
            "ATTRNN (xfacta)":    0.47,
            "ATTRNN (weibo)":     0.20,
            "ATTRNN (mmhl)":      0.00,
            "ATTRNN (snopes)":    0.00,
        }
        # Length-aware routing knobs
        self.short_text_max_words       = 15
        self.ultra_short_max_words      = 8
        self.ultra_short_min_agreement  = 0.68
        # Separate reliability profile for short text (defaults to global until calibrated).
        self.expert_reliability_short = dict(self.expert_reliability)

        self.label_order_map = {
            "MoPeD (mmhl)":      "FR",
            "COOLANT (xfacta)":  "FR",
            "COOLANT (mmhl)":    "FR",
            "EMAF (mmhl)":       "FR",
            "MCAN (mmhl)":       "FR",
            "SpotFake (xfacta)": "FR",
            "MVAE (xfacta)":     "FR",
            "ATTRNN (xfacta)":   "FR",
        }
        for label in self.expert_reliability.keys():
            if label not in self.label_order_map:
                self.label_order_map[label] = "RF"

        # Abstain / uncertainty thresholds 
        self.min_vote_strength = 0.20
        self.min_agreement     = 0.65

        print("All experts loaded and ready.")

    # Language routing
    def _detect_language(self, text):
        """Detect Chinese characters → 'zh', otherwise → 'en'."""
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        return 'en'

    def _iter_experts(self, lang, text=""):
        word_count       = len(text.split())
        is_short_caption = word_count < self.short_text_max_words
        reliability_map  = self.expert_reliability_short if is_short_caption else self.expert_reliability

        # Here is the core of our routing logic: we adjust weights based on language, domain, and reliability.
        if lang == 'zh':
            domain_weight = {"weibo": 1.00, "xfacta": 0.40, "snopes": 0.40, "mmhl": 0.40}
        else:
            domain_weight = {"weibo": 0.35, "xfacta": 1.00, "snopes": 1.00, "mmhl": 1.00}
            # Route based on text length — short captions vs article-style text
            if is_short_caption:
                domain_weight["xfacta"] = 1.20  # xfacta trained on social media style
                domain_weight["snopes"] = 0.70  # snopes is article-focused
            else:
                domain_weight["snopes"] = 1.20  # snopes better for longer news text
                domain_weight["xfacta"] = 0.70

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
                    domain_weight.get(name, 1.0) *
                    reliability_map.get(label, self.expert_reliability.get(label, 1.0))
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

    # Map each expert to a plain-English role for the frontend explanation.
    def _plain_role(self, model_label):
        dataset_roles = {
            "snopes": "fact-checking expert",
            "xfacta": "news verification expert",
            "weibo": "Chinese social media expert",
            "mmhl": "health misinformation expert",
        }
        lower_label = model_label.lower()
        for key, role in dataset_roles.items():
            if key in lower_label:
                return role
        return "multimodal verification expert"

    # Pull out simple text cues the UI can show to the user.
    def _extract_review_cues(self, text):
        lowered = text.lower()
        cues = []

        # Keep cue matches short and remove duplicates.
        def add_cue(label, matches):
            cleaned = []
            for match in matches:
                if match and match not in cleaned:
                    cleaned.append(match)
            if cleaned:
                cues.append({"label": label, "matches": cleaned[:3]})

        health_terms = [
            "covid", "vaccine", "doctor", "medical", "health",
            "virus", "symptom", "smell", "cure", "treatment",
            "新冠", "疫情", "治愈", "医生", "健康"
        ]
        urgency_terms = [
            "breaking", "warning", "urgent", "must", "immediately",
            "shocking", "miracle", "secret", "转发", "求证", "震惊", "紧急"
        ]
        money_terms = [
            "million", "billion", "inherit", "prize", "free", "cash",
            "giveaway", "美元", "亿元", "万", "$"
        ]

        add_cue("Health-related claim", [term for term in health_terms if term in lowered or term in text])
        add_cue("Urgent or emotional wording", [term for term in urgency_terms if term in lowered or term in text])
        add_cue("Money or scale claims", [term for term in money_terms if term in lowered or term in text])

        number_matches = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
        if number_matches:
            add_cue("Notable numbers", number_matches[:3])

        word_count = len(text.split())
        if word_count <= self.ultra_short_max_words:
            add_cue("Very short text", [f"{word_count} words"])

        return cues[:3]

    # Build the extra XAI fields that the frontend shows in the 'simple' view.
    def _build_xai_fields(self, text, all_results, final_verdict, agreement_weight, vote_strength, is_uncertain, uncertainty_reason):
        active = [r for r in all_results if r["weight"] > 0.0]
        supporters = sorted(
            [r for r in active if r["predicted_label"] == final_verdict],
            key=lambda x: x["weight"] * x["margin"],
            reverse=True
        )
        top_supporters = supporters[:3]
        role_map = OrderedDict()
        for row in top_supporters:
            role = self._plain_role(row["model"])
            if role not in role_map:
                role_map[role] = row["model"]

        plain_roles = list(role_map.keys())
        support_count = len(supporters)
        active_count = len(active)

        if agreement_weight >= 0.80:
            agreement_phrase = "high"
        elif agreement_weight >= 0.65:
            agreement_phrase = "moderate"
        else:
            agreement_phrase = "low"

        if vote_strength >= 0.40:
            strength_phrase = "strong"
        elif vote_strength >= 0.25:
            strength_phrase = "moderate"
        else:
            strength_phrase = "narrow"

        if is_uncertain:
            summary = "The experts did not align strongly enough to make a confident decision."
        elif final_verdict == "Fake":
            summary = f"Most active experts leaned Fake, with {agreement_phrase} agreement and a {strength_phrase} separation between the Real and Fake votes."
        else:
            summary = f"Most active experts leaned Real, with {agreement_phrase} agreement and a {strength_phrase} separation between the Real and Fake votes."

        next_steps = [
            "Search the claim on fact-check sites.",
            "Look for the original source of the post or image.",
            "Treat this as a pattern-based AI signal, not proof."
        ]
        if uncertainty_reason:
            next_steps[0] = "Look for the original source before trusting the claim."

        return {
            "xai_plain_summary": summary,
            "xai_plain_roles": plain_roles,
            "xai_support_count": support_count,
            "xai_active_count": active_count,
            "xai_review_cues": self._extract_review_cues(text),
            "xai_next_steps": next_steps,
        }

    # Core prediction
    def get_prediction(self, text, image_path=None):
        # Queries active experts and performs weighted soft voting
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` is required and must be a non-empty string.")
        if not self._has_real_image_input(image_path):
            raise ValueError("`image_path` is required — this ensemble expects both text and image.")

        img  = self._resolve_input_image(image_path)
        lang = self._detect_language(text)
        word_count = len(text.split())
        is_short_text = word_count < self.short_text_max_words
        is_ultra_short = word_count <= self.ultra_short_max_words

        all_results   = []
        weighted_real = 0.0
        weighted_fake = 0.0
        total_weight  = 0.0

        for model_label, expert, weight in self._iter_experts(lang, text):
            res = expert.predict(text, img)
            real_p, fake_p = self._normalize_probs(res['Real'], res['Fake'])
            real_p, fake_p = self._apply_label_order(model_label, real_p, fake_p)
            margin = abs(real_p - fake_p)

            all_results.append({
                "model": model_label,
                "Real": real_p,
                "Fake": fake_p,
                "weight": float(weight),
                "margin": margin,
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

        # Calculate base confidence 
        confidence = 0.5 + 0.5 * (vote_strength * agreement_weight)

        # Applying the anchor boost logic
        if lang == "zh":
            anchor_models = []  # no Chinese-domain experts in the demo
        elif is_short_text:
            anchor_models = ["SpotFake (xfacta)", "MVAE (xfacta)"]
        else:
            anchor_models = ["MVAE (xfacta)", "ATTRNN (xfacta)"]

        anchor_votes = [r for r in all_results if r["model"] in anchor_models]
        anchor_unanimous = len(anchor_votes) > 0 and all(
            r["predicted_label"] == final_verdict for r in anchor_votes
        )
        
        # We need to avoid boosting confidence for ultra-short text where info for the text is limited.
        if anchor_unanimous and not is_ultra_short:
            confidence = min(0.99, confidence * 1.10)

        # Find out the uncertainty 
        ultra_short_low_agreement = is_ultra_short and (agreement_weight < self.ultra_short_min_agreement)
        is_uncertain = (
            (vote_strength < self.min_vote_strength)
            or (agreement_weight < self.min_agreement)
            or ultra_short_low_agreement
        )

        uncertainty_reason = None
        if is_uncertain:
            if ultra_short_low_agreement:
                uncertainty_reason = "ultra_short_low_agreement"
            elif vote_strength < self.min_vote_strength and agreement_weight < self.min_agreement:
                uncertainty_reason = "low_vote_strength_and_low_agreement"
            elif vote_strength < self.min_vote_strength:
                uncertainty_reason = "low_vote_strength"
            else:
                uncertainty_reason = "low_agreement"

        best_overall = max(all_results, key=lambda x: x["margin"])

        # Lets package the extra explanation data once so both frontends can reuse it.
        xai_fields = self._build_xai_fields(
            text,
            all_results,
            final_verdict,
            agreement_weight,
            vote_strength,
            is_uncertain,
            uncertainty_reason,
        )

        return {
            "final_verdict": "Uncertain" if is_uncertain else final_verdict,
            "overall_confidence":confidence,
            "most_confident_model": best_overall['model'],
            "best_model_score":best_overall["margin"],
            "language_detected":lang,
            "used_real_image":True,
            "text_word_count": word_count,
            "used_short_profile": is_short_text,
            "vote_strength": vote_strength,
            "agreement": agreement_weight,
            "is_uncertain": is_uncertain,
            "uncertainty_reason": uncertainty_reason,
            "avg_real": avg_real,
            "avg_fake": avg_fake,
            "all_scores": all_results,
            **xai_fields,
        }

    # Batch benchmark
    # It will be important to run `fit_label_order_from_benchmark` on any new batch of cases before interpreting these results, 
    # to make sure that label-order corrections are properly applied
    def benchmark_prompts(self, cases):
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
                "used_real_image":pred["used_real_image"],
                "vote_strength":  pred["vote_strength"],
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

    # This function checks benchmark cases with known labels to see if any expert is using reversed class outputs
    # It is called during evaluation before interpreting benchmark results
    def fit_label_order_from_benchmark(self, cases, min_cases=3, min_improvement=0.15, auto_apply=True):
        labeled_cases = [c for c in cases if c.get("expected") in ("Real", "Fake")]
        if len(labeled_cases) < min_cases:
            raise ValueError(f"Need at least {min_cases} labeled cases to fit label order.")

        prepared = [{
            "text": c["text"],
            "expected": c["expected"],
            "img": self._resolve_input_image(c.get("image_path")),
        } for c in labeled_cases]

        report = {}
        for model_label, expert in self._all_experts_flat().items():
            normal_correct  = 0
            flipped_correct = 0
            for case in prepared:
                res = expert.predict(case["text"], case["img"])
                real_p, fake_p = self._normalize_probs(res["Real"], res["Fake"])
                if ("Fake" if fake_p > real_p else "Real") == case["expected"]: normal_correct  += 1
                if ("Fake" if real_p > fake_p else "Real") == case["expected"]: flipped_correct += 1

            n = len(prepared)
            normal_acc = normal_correct  / n
            flipped_acc = flipped_correct / n
            improvement = flipped_acc - normal_acc
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

    # Lets automatically compute reliability weights from benchmark results, based on accuracy on labeled cases. 
    # This can be used to update `self.expert_reliability` after running a benchmark with known labels, to better weight the experts in future predictions.
    def compute_reliability_from_benchmark(self, cases, target_map="global"):
        labeled = [c for c in cases if c.get("expected") in ("Real", "Fake")]
        if len(labeled) < 10:
            raise ValueError("Need at least 10 labeled cases to compute reliability.")
        if target_map not in ("global", "short"):
            raise ValueError("target_map must be 'global' or 'short'.")

        prepared = [{
            "text":     c["text"],
            "expected": c["expected"],
            "img":      self._resolve_input_image(c.get("image_path")),
        } for c in labeled]

        title = "New Weight (short)" if target_map == "short" else "New Weight"
        print(f"\n{'Model':<25} {'Accuracy':>9} {title:>17}")
        print("-" * 48)

        for model_label, expert in self._all_experts_flat().items():
            correct = 0
            for case in prepared:
                res            = expert.predict(case["text"], case["img"])
                real_p, fake_p = self._normalize_probs(res["Real"], res["Fake"])
                real_p, fake_p = self._apply_label_order(model_label, real_p, fake_p)
                predicted      = "Fake" if fake_p > real_p else "Real"
                if predicted == case["expected"]:
                    correct += 1

            accuracy   = correct / len(prepared)
            new_weight = max(0.0, (accuracy - 0.5) * 2)
            if target_map == "short":
                self.expert_reliability_short[model_label] = new_weight
            else:
                self.expert_reliability[model_label] = new_weight
            print(f"  {model_label:<23} {accuracy:>8.0%} {new_weight:>16.2f}")

    def compute_short_text_reliability_from_benchmark(self, cases, max_words=None, min_cases=10):
        max_words = self.short_text_max_words if max_words is None else int(max_words)
        short_labeled = [
            c for c in cases
            if c.get("expected") in ("Real", "Fake")
            and len(str(c.get("text", "")).split()) < max_words
        ]
        if len(short_labeled) < min_cases:
            raise ValueError(
                f"Need at least {min_cases} short labeled cases (<{max_words} words). "
                f"Found {len(short_labeled)}."
            )
        self.compute_reliability_from_benchmark(short_labeled, target_map="short")
        return {
            "target_map": "short",
            "max_words": max_words,
            "num_cases": len(short_labeled),
        }

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
