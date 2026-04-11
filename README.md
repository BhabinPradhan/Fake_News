# MOSAIC — Cross-Domain Multimodal Fake News Detection

> University of Windsor COMP 4990 4th Year Undergraduate Capstone Project

## What It Does
Our MOSAIC program takes a social media post (a text and image) and classifies it as either **Real**, **Fake**, or **Uncertain**.
It works by running the input through an ensemble of 28 expert models spreading across 7 different model families
and 4 dataset domains, which we then combine their votes using weighted ensemble logic. 
Rather than forcing a falsely confident guess on inputs that might seem ambiguous, the system returns `Uncertain` when vote strength or
agreement falls below a confidence threshold. This is an intentional feature that was added, not a failure.

Keep note that MOSAIC is a pattern-based detector trained on misinformation datasets. It is not a live
fact-checker and does not verify claims against external sources in real time.

## System Architecture
```text
The User's Browser
    │
    ▼
index.html  (frontend; hosted on the University of Windsor MyWeb)
    │
    │  fetch() to the Flask backend through an ngrok tunnel
    ▼
api.py  (Flask backend; /health, /predict, /scrape)
    │
    ▼
model_manager.py  (ensemble inference engine)
    │
    ├── Domain & language routing
    ├── 28 expert models (7 families × 4 dataset domains)
    ├── Weighted ensemble voting
    │     final weight = domain weight × reliability weight
    └── XAI field generation for frontend display
          plain-language summary for non-technical users
          technical expert breakdown and supporting expert list
          review cues and external fact-check links
```

Key thresholds in the ensemble:
- `min_vote_strength (controls how large the ensemble margin must be) = 0.20`
- `min_agreement (controls how many active experts must align) = 0.65`
- `ultra_short_min_agreement (is a stricter agreement rule for very short inputs) = 0.68`

---

## Project Structure
Our repository is organized around the ensemble inference engine, frontend/demo interfaces, benchmarking utilities, and model-specific wrappers. Some expert families are stored in dedicated folders, while others are loaded through wrapper modules in the project root.

```
mosaic/
├── model_manager.py              # Core ensemble engine: routing, weighting, inference, and XAI generation
├── api.py                        # Flask backend exposing /health, /predict, and /scrape
├── scraper.py                    # URL scraping helper used by the backend
├── index.html                    # Main public-facing web interface
├── mosaic_app.py                 # Streamlit-based demo interface
│
├── benchmark_loader.py           # Loads balanced multimodal benchmark samples
├── benchmark_seed_sweep.py       # Repeats benchmarks across random seeds
├── benchmark_exhaustive_eval.py  # Runs multi-size, multi-seed evaluation
│
├── results/                      # Final benchmark CSV outputs
├── multimodal_dataset/           # Dataset splits and paired evaluation samples
├── weights/                      # Trained model weight files
│
├── MoPeD/                        # MoPeD model files
├── EMAF/                         # EMAF architecture files
├── COOLANT/                      # COOLANT architecture files
│
├── *_wrapper.py                  # Expert model wrapper modules
├── requirements.txt              # Python dependency list
│
└── docs/
    ├── system_design.md          # Architecture, ensemble logic, and routing design
    ├── api_reference.md          # Endpoint specifications and usage examples
    ├── results.md                # Benchmark methodology, results, and limitations
    └── user_guide.md             # Frontend usage and demo setup instructions
```
## Setup & Installation
<!-- Amer or shared: Python version, pip install, how to run api.py -->

## Running the Demo
<!-- Bhabin: ngrok command, local test path, what to expect -->

## API Endpoints
<!-- Marc: /health, /predict, /scrape with examples -->

## Benchmark Results
<!-- Darren: summary of 100-case aggregate, point to CSV files -->

## Known Limitations
<!-- Amer: paste/adapt from ProjectSummary.txt limitations section -->

## Team
<!-- Everyone: names and roles -->
