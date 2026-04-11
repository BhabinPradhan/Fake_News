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
<!-- Amer: annotated file tree of core files -->

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
