# System Design

## 1. System Overview

MOSAIC is a cross-domain multimodal fake news detection system that takes paired text and image input and classifies it as `Real`, `Fake`, or `Uncertain`. Now instead of just relying on a single model trained on one dataset, MOSAIC combines 28 expert models spread across 7 model families and 4 dataset domains. This ensemble design was chosen because misinformation appears in different styles across platforms and topics like short social captions, article-style claims, health misinformation, and Chinese social media posts that do not all look alike. The ensemble allows the system to reuse domain-specialized experts, re-weight them based on information from input, and abstain when the vote is too weak or too conflicted.

## 2. High-Level Architecture

The system has three main important layers: frontend, a Flask backend, and the ensemble inference system

```text
The User's Browser
    │
    ▼
index.html / index_local_test.html
    │
    │  fetch() requests
    ▼
api.py
    │
    ▼
model_manager.py
    │
    ├── Language and routing logic
    ├── 28 expert models
    ├── Weighted soft voting
    └── XAI field generation
```

- `index.html` is the public/demo frontend used with the setup.
- `index_local_test.html` is the local testing frontend configured for `http://127.0.0.1:5000`.
- `api.py` exposes three endpoints:
  - `/health` for backend status check
  - `/predict` for multimodal inference
  - `/scrape` for extracting article text and an image from a URL
- `model_manager.py` loads all expert models, applies routing and weighting logic, performs ensemble voting, and generates the explanation fields used by the frontend.

The backend keeps a single global `ModelManager` instance alive after startup. This avoids reloading all 28 experts on every request...

## 3. Expert Model Families and Dataset Domains

The ensemble contains 7 model families:

- MoPeD
- COOLANT
- EMAF
- MCAN
- SpotFake
- MVAE
- ATTRNN

Each family contributes one expert for each of 4 domains, giving a 7 × 4 matrix of 28 experts:

- `xfacta`: English cross-domain news verification
- `snopes + fakeddit`: English fact-check and article-style claim verification with fakeddit as a subset (keyed as snopes in code)
- `weibo`: Chinese social media misinformation
- `mmhl`: Health misinformation

This design lets MOSAIC reuse different types of pretrained experts instead of assuming one data source is for of every misinformation setting. In practice:

- `xfacta` experts are useful for shorter, social-style captions and posts.
- `snopes + fakeddit` experts are better for longer, article-like news claims.
- `weibo` experts are increased when Chinese text is found.
- `mmhl` experts add health-domain coverage when medical or public-health style misinformation appears.

These domains were chosen because they span across different linguistic, topical, and stylistic conditions that a single benchmark alone would not capture well.

## 4. Ensemble Weighting Logic

The core idea and purpose of MOSAIC is the weighted ensemble voting. Each expert contributes a soft prediction consisting of `Real` and `Fake` probabilities. Those outputs are normalized, some were corrected for label order, and then combined using a per-expert final weight:

```text
final_weight = domain_weight × reliability_weight
```

### 4.1 Domain Weights

`domain_weight` is determined dynamically inside `_iter_experts(lang, text)`. It looks at how relevant a dataset domain is for the current input.

For Chinese input:

```text
weibo  = 1.00
xfacta = 0.40
snopes = 0.40
mmhl   = 0.40
```

For English input, the default profile is:

```text
weibo  = 0.35
xfacta = 1.00
snopes = 1.00
mmhl   = 1.00
```

English routing is then adjusted by text length:

- If the text is short (`word_count < 15`):
  - `xfacta = 1.20`
  - `snopes = 0.70`
- If not:
  - `snopes = 1.20`
  - `xfacta = 0.70`

This lets the ensemble lean toward social-caption experts for short posts and article-trained experts for longer claims.

### 4.2 Reliability Weights

`reliability_weight` comes from the calibrated `expert_reliability` map stored in `ModelManager`. These values look how much trust the ensemble places in each expert based on benchmark performance. Stronger experts receive larger weights, while weaker experts are weighted negatively. Experts with a reliability of `0.00` are effectively removed from voting because their final weight becomes zero.

The code also keeps a separate `expert_reliability_short` for short-text routing. At the moment it is initialized from the global reliability map, but the class includes helper functions such as `compute_short_text_reliability_from_benchmark()` so this can be calibrated separately for any future experiments.

### 4.3 Label-Order Correction

Not all wrapped expert models show their outputs in the same class order. Some return `[Real, Fake]` while others return `[Fake, Real]`, depending on how they were originally trained. Averaging across them directly would silently ruin the whole ensemble vote.

MOSAIC handles this with a label_order_map that tags each expert as either RF (no change needed) or FR (probabilities swapped before ensembling). The map is hard-coded based on per-wrapper verification. The class also includes `fit_label_order_from_benchmark()`, which was used during development to see and confirm the correct assignment for each expert against labeled cases. This was useful when a wrapper's output order is ambiguous or undocumented.

### 4.4 Weighted Voting

For each active expert, MOSAIC records:

- normalized `Real` probability
- normalized `Fake` probability
- corrected label order
- final ensemble weight
- prediction margin

The final weighted vote is then computed as:

```text
avg_real = Σ(real_i × weight_i) / Σ(weight_i)
avg_fake = Σ(fake_i × weight_i) / Σ(weight_i)
```

The initial verdict is:

- `Fake` if `avg_fake > avg_real`
- `Real` otherwise

Two summary stats are then found:

- `vote_strength = |avg_fake - avg_real|`
- `agreement = weighted proportion of experts on the winning side`

### 4.5 Confidence and Anchor Boost

Base confidence is computed as:

```text
confidence = 0.5 + 0.5 × (vote_strength × agreement)
```

The system then applies an 'anchor boost' when a small set of important anchor experts all support the same final verdict:

- Chinese input uses Weibo anchors:
  - `MoPeD (weibo)`
  - `COOLANT (weibo)`
  - `MCAN (weibo)`
- Short English text uses XFacta anchors:
  - `MoPeD (xfacta)`
  - `COOLANT (xfacta)`
  - `MCAN (xfacta)`
- Longer English text uses Snopes anchors:
  - `MoPeD (snopes)`
  - `COOLANT (snopes)`
  - `MCAN (snopes)`

If these anchors are unanimous and the sample is not ultra-short, confidence is multiplied by `1.10` and capped at `0.99`.

### 4.6 The `Uncertain` Output

MOSAIC is allowed to say "Im unsure" instead of forcing a prediction. This is implemented in `get_prediction()` through three thresholds:

- `min_vote_strength = 0.20`
- `min_agreement = 0.65`
- `ultra_short_min_agreement = 0.68`

An output becomes `Uncertain` if any of these conditions happen:

- `vote_strength < min_vote_strength`
- `agreement < min_agreement`
- the text is ultra-short (`word_count <= 8`) and `agreement < ultra_short_min_agreement`

The system also records a structured `uncertainty_reason`, such as:

- `low_vote_strength`
- `low_agreement`
- `low_vote_strength_and_low_agreement`
- `ultra_short_low_agreement`

This abstention mechanism is intentional and one of the core ideas of MOSAIC. It is designed to improve trustworthiness by preventing weak or conflicting votes from being shown as confident classifications!

## 5. Routing Logic

Routing in MOSAIC is weight-based rather than hard model selection. The system does not choose one expert family and ignore the rest. Instead, it keeps any expert whose final weight is greater than zero and changes that influence through routing.

### 5.1 Language Routing

Language routing is handled by `_detect_language(text)`, which checks for Chinese Unicode characters:

- Chinese characters detected: `zh`
- otherwise: `en`

This directly affects the domain-weight profile inside `_iter_experts()`, especially the emphasis on the Weibo experts for Chinese input.

### 5.2 Length-Aware Routing

The system also routes by text length:

- short text: `word_count < 15`
- ultra-short text: `word_count <= 8`

Short-text routing shifts more weight toward `xfacta`, since those experts were trained on shorter social-style content. Longer text shifts more weight toward `snopes`, which better matches article-style claims.

Ultra-short text is treated even more cautiously. Instead of relying only on the normal agreement threshold, MOSAIC applies the stricter `ultra_short_min_agreement` threshold because very short text often carries too little information for multimodal classification.

### 5.3 Chinese and Weibo Routing

Chinese routing into the Weibo path was checked during development. The current implementation increases Weibo influence for Chinese input while still allowing lower-weight contributions from other domains.

## 6. The Inference Pipeline

When `get_prediction(text, image_path)` is called, the pipeline runs as follows:

1. Check that the text is non-empty and that a real image was provided.
2. Resolve the image into a PIL RGB image.
3. Detect the input language using `_detect_language(text)`.
4. Count words and see whether the sample is short or ultra-short.
5. Iterate through active experts using `_iter_experts(lang, text)`.
6. For each expert:
   - run `expert.predict(text, img)`
   - normalize `Real` and `Fake` scores
   - apply label-order correction if needed
   - compute prediction margin
   - append the result to `all_scores`
7. Combine all expert scores using weighted soft voting.
8. Determine the preliminary `Real` or `Fake` verdict.
9. Compute `vote_strength`, `agreement`, and base confidence.
10. Apply anchor boost if anchor experts all support the same verdict and the text is not ultra-short.
11. Check abstention thresholds and convert the prediction to `Uncertain` if needed.
12. Build explanation with `_build_xai_fields(...)`.
13. Return the final response object, including:
   - `final_verdict`
   - `overall_confidence`
   - `vote_strength`
   - `agreement`
   - `language_detected`
   - `most_confident_model`
   - `all_scores`
   - XAI fields for the frontend

## 7. XAI Layer

MOSAIC includes an explainability layer so the frontend can show more than just a 'label' that people might not trust.

The backend helper `_build_xai_fields()` produces:

- `xai_plain_summary`
- `xai_plain_roles`
- `xai_support_count`
- `xai_active_count`
- `xai_review_cues`
- `xai_next_steps`

These fields are lightweight and user-facing intentionally. They aren't meant to be actual formal proof.... Instead, they summarize:

- which kinds of experts supported the result
- what kinds of cues appeared in the text
- what a human reviewer should check next

The frontend then presents this explanation in two views:

- **Simple View** for non-technical users
- **Technical View** showcases per-expert support and opposition, dataset hints, and vote snapshot details

This split was added so the interface could be understood for normal users while still showing enough detail for debugging, demonstration, and technical review.

## 8. Some Key Design Decisions

### 8.1 Why an Ensemble Instead of One Model

Misinformation can vary a lot by topic, language, and platform style. Because of this, a single model trained on one dataset wouldn't have great domain knowledge (a common problem in ML!). The ensemble approach lets MOSAIC reuse domain-specialized experts and combine them in a more helpful way.

### 8.2 Why `Uncertain` Is Better Than Forced Prediction

Many misinformation systems are looked at as if every case must be labeled. In many real life scenarios, weak or conflicting evidence is common. As a result, MOSAIC treats abstention as a safety feature. Returning `Uncertain` is preferable to making a confident-looking but poorly supported guess.

### 8.3 Why Reliability Weights Were Benchmark-Driven

The our map for reliability isnt just a guessed preference list. The values are from benchmarked calibrations. Some experts performed well in certain domains, while others added useless noise. Reliability weighting is smart in this case as it lets the ensemble keep strong domain specialists and silence weak contributors without removing the overall cross-domain structure.

### 8.4 Why the Weibo Path Was Patched Instead of Retrained

In our submission, we decided to focus on integration and the behavior of the ensemble; not full retraining of every expert family due to a lack of time. Because routing issues with the Chinese data were somewhat caused by weight calibration and output-order mismatch (0=Fake or Real), patching the Weibo path was a justified solution for the project.

## 9. Training Artifacts and Wrapper Integration

Each of the 7 model families came from a separate research implementation. They do not all share a common interface. Some of the models expose a single predict() call, others need a dataset-specific preprocessing, and some also output probabilities in a different class order depending on how they were trained. Making sure to get all of them to behave consistently inside a single ensemble needed a dedicated wrapper for each family and domain combination.

The wrapper files (`moped_wrapper.py`, `coolant_wrapper.py`, `emaf_wrapper.py`, and the dataset-specific variants for SpotFake, MVAE, and ATTRNN) handle the differences in preprocessing, tokenization, and output format before anything reaches model_manager.py. By the time `_iter_experts()` shows an expert, it can be they call all be treated equally regardless of which architecture it wraps.

This 'normalization work' is why model_manager.py is structured in a way to include an integration layer rather than a model itself. The core ensemble logic looks and stays clean because the messiness of interacting with 7 different codebases is taken in at the wrapper level.

## 10. Known Constraints and Deployment Notes

### 10.1 Multi-GPU Deployment

The original deployment distributes experts across multiple CUDA devices:

- `cuda:0`
- `cuda:1`
- `cuda:2`
- `cuda:3`

This was done to reduce memory pressure and make the full ensemble run smoothly on the university server. People with fewer GPUs may need to edit device assignments in `model_manager.py`. 

### 10.2 Backend and Frontend Deployment

The backend is served by Flask through `api.py` on port `5000`.

- local testing uses `index_local_test.html`
- the public/demo setup uses `index.html`
- the deployed demo path used ngrok to expose the backend to the public-facing frontend

This separation prevents local testing changes from breaking the public demo configuration.

### 10.3 Scraping Constraints

The `/scrape` endpoint relies on `scraper.py`, which uses Playwright and `newspaper3k` for article extraction, with platform-specific handling for some sites. Scraping can be fragile in some cases:

- some sites block automated access
- live-update pages can produce noisy text
- some pages do not expose a usable lead image

For this reason, scraped inputs are treated as a convenience feature.

### 10.4 Scope Limits

The repository does doesn't include a full retraining pipeline for all expert families. Training scripts like MoPeD_weibo.py and emaf_snopes.py are not included. The original model implementations are listed in the references section. The submission focuses on the ensemble layer which includes routing, weighting, abstention logic, and the inference flow is all built around those pretrained experts. Retraining all seven families from scratch was outside the scope with the timing we had, and the integration work was substantial enough on its own.
