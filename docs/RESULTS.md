# RESULTS & EVALUATION REPORT

## Benchmark Methodology

**Benchmark Design**
- balanced sampling from `multimodal_dataset/test_final.csv`
- only original paired image-text samples were included
- repeated evaluation across multiple random seeds
- benchmark sizes of 60, 80, and 100 cases
- results generated using `benchmark_seed_sweep.py` and `benchmark_exhaustive_eval.py`
- accuracy reported on decided cases only
- coverage and uncertain rate reported separately

The benchmark was built using balanced sampling from `multimodal_dataset/test_final.csv` to maintain equal class representation during evaluation. Only original paired image-text samples were used so that the multimodal relationship between text and image remained intact. Evaluations were carried out at benchmark sizes of 60, 80, and 100 samples and were repeated across multiple random seeds to reduce sampling bias. All reported results were generated using `benchmark_seed_sweep.py` and `benchmark_exhaustive_eval.py`, and no additional reruns were performed after the results were frozen.

Accuracy is reported **only on decided cases**, meaning inputs where MOSAIC predicted either *Real* or *Fake*. When the system does not reach a strong enough internal signal, it returns *Uncertain* instead of forcing a classification. For that reason, coverage and uncertain rate are reported separately so that performance is not overstated.

**Metric Definitions**
- decided accuracy = correct predictions divided by decided cases only
- coverage = percentage of inputs receiving a Real or Fake decision
- uncertain rate = percentage of inputs labeled Uncertain
- Real/Fake accuracy = class-wise accuracy on decided cases

Decided accuracy measures correctness only when the system commits to a prediction. Coverage reflects how often MOSAIC produces a Real or Fake output, while uncertain rate captures intentional abstentions. Real and Fake accuracy measure class-wise performance on decided cases only.

---

## Final Results

**Aggregate Benchmark Outcomes**
- results averaged across repeated seed evaluations
- evaluated at 60, 80, and 100 cases
- reported metrics follow the benchmark definitions above
- final validated claim based on repeated English multimodal evaluation

| Cases | Mean Decided Accuracy | Standard Deviation | Coverage | Uncertain Rate | Real Accuracy | Fake Accuracy |
|------|-----------------------|--------------------|----------|----------------|---------------|---------------|
| 60   | 88.77% | 5.53% | 53.67% | 46.33% | 91.00% | 86.85% |
| 80   | 89.11% | 3.86% | 53.50% | 46.50% | 92.52% | 85.98% |
| 100  | 89.46% | 3.60% | 54.80% | 45.20% | 91.25% | 87.63% |

Results were aggregated across multiple seed runs rather than taken from a single evaluation. Across benchmark sizes, decided accuracy converged near **89.5%**, which serves as the main reported performance figure. However, this value refers **only to decided cases**, so it must be interpreted together with coverage and uncertain rate. The repeated English multimodal benchmark therefore represents the validated final result, rather than any one individual run.

---

## Stability Across Seeds

**Seed Sweep Evaluation**
- each benchmark size evaluated across multiple random seeds
- per-run results reviewed using `benchmark_eval_per_run.csv`
- variance checked to reduce the chance of sample-specific conclusions
- standard deviation decreased as benchmark size increased

Evaluation was repeated across seeds to ensure that the reported results were not dependent on one favourable sample. The per-run outputs show that performance remained reasonably consistent across runs. In addition, the reduction in standard deviation from smaller to larger benchmark sizes suggests that the larger evaluations provide more statistically reliable estimates of performance. This suggests that the results are stable, not just a lucky sample.

---

## Interpretation

**System Behavior**
- MOSAIC is a multimodal misinformation detection system, not a fact-checking system
- abstention through *Uncertain* is intentional
- reliability is prioritized over full coverage
- accuracy must be interpreted alongside coverage and uncertain rate

MOSAIC detects multimodal patterns associated with misinformation rather than verifying factual claims directly. The inclusion of an *Uncertain* output is intentional and allows the system to avoid forced classifications when expert signals conflict or confidence is too low. As a result, decided accuracy shows performance on confident predictions, while coverage shows how often the system commits to a decision. These values must be interpreted together to give a fair picture of behaviour.

---

## Chinese Limitation (Weibo16)

**Chinese Evaluation Findings**
- routing into Chinese / Weibo experts was verified
- the system correctly directed Chinese inputs to the Weibo path
- end-to-end Chinese evaluation showed strong class imbalance
- Chinese performance was excluded from the primary benchmark claims
- Chinese evaluation is documented as a limitation and future work

Routing into Chinese and Weibo-specific expert models was verified to function correctly. However, end-to-end Chinese evaluation showed strong class imbalance, which made Chinese performance unsuitable for use in the main validated claims. Because of this, the final reported conclusions will focus on the English multimodal benchmark, while Chinese evaluation is marked and documented as a known limitation and an area for future work.
