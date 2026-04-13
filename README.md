# MOSAIC — Cross-Domain Multimodal Fake News Detection

> University of Windsor COMP 4990 4th Year Undergraduate Capstone Project

## What It Does
Our MOSAIC program takes a social media post (a text and image) and classifies it as either **Real**, **Fake**, or **Uncertain**.
It works by running the input through an ensemble of 28 expert models spread across 7 different models
and 4 datasets, and combines their votes using weighted ensemble logic. 

Rather than forcing a falsely confident guess on inputs that might seem ambiguous, the system returns `Uncertain` when vote strength or
agreement falls below a confidence threshold. This is an intentional feature that was added, not a failure.

Keep note that MOSAIC is a pattern-based detector trained on misinformation datasets. It is not a live
fact-checker and does not verify claims against external sources in real time.

## System Architecture

During development, expert models were distributed across multiple available CUDA devices on the university's SSH server to reduce memeory load and support full-ensemble inference.

```text
The User's Browser
    │
    ▼
index.html (the public/demo frontend)
    │
    │  fetch() to the Flask backend
    ▼
api.py (Flask backend; /health, /predict, /scrape)
    │
    ▼
model_manager.py (the ensemble inference engine)
    │
    ├── Domain & language routing
    ├── 28 expert models (7 families × 4 dataset domains)
    ├── Weighted ensemble voting
    │   └── final_weight = domain_weight × reliability_weight
    └── XAI field generation for frontend display
        ├── Plain-language summary for users that are non technical 
        ├── Technical expert breakdown and supporting expert list
        └── Review cues and external fact-check links
```

The system primarily uses `index.html` for the public-facing demo. For local development and for validation, `index_local_test.html` is used as the dedicated local testing frontend.

Key thresholds in the ensemble:
- `min_vote_strength = 0.20`  
  Controls how large the ensemble vote margin must be before a prediction is accepted.
- `min_agreement = 0.65`  
  Controls how many active experts must agree before the system commits to a prediction.
- `ultra_short_min_agreement = 0.68`  
  Applies a stricter agreement rule for very short inputs.

---

## Project Structure
Our repository is organized around the ensemble inference engine, frontend/demo interfaces, benchmarking utilities, and model-specific wrappers. Some expert families are stored in dedicated folders, while others are loaded through wrapper modules in the project root.

```
mosaic/
├── model_manager.py              # Core ensemble engine: routing, weighting, inference, and XAI generation
├── api.py                        # Flask backend exposing /health, /predict, and /scrape
├── scraper.py                    # URL scraping helper used by the backend
├── index.html                    # Main web frontend
├── index_local_test.html         # Local frontend for testing without ngrok
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
> **Note:** This project was developed on the University of Windsor Delta GPU server using a Conda environment. In that environment, `pip install` and `sudo` access were restricted, so dependencies were installed manually through Conda. The included `requirements.txt` documents the main Python dependencies, but the exact setup may need to be changed to your environment.

### Requirements
- Python 3.13 recommended
- Conda or another Python virtual environment
- GPU access highly recommended for faster inference speed

> The original deployment distributed expert models across multiple CUDA devices on the university server. If your system has fewer GPUs, you may need to update the device assignments in model_manager.py, and CPU-only execution will be significantly slower.

### 1. Clone the Repository

```bash
git clone <repo-url>
cd mosaic
```
### 2. Create the Environment
Using Conda (recommended):
```bash
conda create -n mosaic python=3.13
conda activate mosaic
```
Install the required packages using requirements.txt as a reference for the environment setup.

### 3. Download the Pretrained Weights
The pretrained model weights are not included in the repository due to file size limits. Download the weights from the shared Google Drive folder provided with the submission
and place all files into the weights/ folder. Do not rename or reorganize the files. Estimated file size for all .pth files ~25GB
```
mosaic/
├── weights/
    ├── spotfake_snopes.pth
    ├── mvae_snopes.pth
    ...

```
> If the required weight files are missing or placed incorrectly, model_manager.py will not be able to load all expert models!

### 4. Run the Backend
First, start the Flask backend by running:
```bash
python api.py
```
The Flask backend will start on `http://127.0.0.1:5000`. You can verify that it is running with:
```bash
curl http://127.0.0.1:5000/health
```

### 5. Local Frontend Testing

For local testing, an ngrok setup is not required.

Open two terminals and make sure both commands are run from the project root (Let's use `Example_Folder/` as an example).

1. Start the Flask backend:

```bash
cd /path/to/Example_Folder
python api.py
```
2. Serve the frontend locally:

```bash
cd /path/to/Example_Folder
python -m http.server 8000
```

3. Open `index_local_test.html` in your browser through the local server.
For example:

```text
http://127.0.0.1:8000/index_local_test.html
```

Use `index_local_test.html` for local testing so that the public/demo `index.html` file does not need to be edited.

### 6. Optional Public Demo Setup

This step is only needed if you want to expose the backend through the public-facing demo frontend instead of testing locally.

1. Start the Flask backend:

```bash
python api.py
```

2. Install and authenticate ngrok using your own ngrok account.

3. Start an ngrok tunnel for port `5000`:

```bash
./ngrok http 5000
```

If you are using a reserved ngrok domain, you can run:

```bash
./ngrok http --url=<your-ngrok-domain> 5000
```

4. Update the `API_BASE` value in `index.html` to match the ngrok URL.

5. If you are serving the frontend locally instead of using MyWeb, start a simple static server:

```bash
python -m http.server 8000
```

Then open the frontend in your browser from the local server.

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Verifies backend status and ensemble model readiness |
| `/predict` | `POST` | Primary detection endpoint (Text + Image) |
| `/scrape` | `POST` | Automated URL extraction and classification |

> For full request/response JSON specifications, example `curl` commands, and scraping platform support, see the [Detailed API Reference](docs/api_reference.md).


## Benchmark Results
MOSAIC was evaluated on a balanced multimodal benchmark using paired text and image samples, with repeated evaluation across multiple random seeds to ensure stable results. At the 100-case benchmark size, the system achieved approximately **89.5% decided accuracy** with about **54.8% coverage**, meaning predictions are only made when confidence is sufficient. Accuracy is reported on decided cases only, while uncertain cases are intentionally abstained from to improve reliability. For the full evaluation methodology, metrics, and detailed results, see [docs/RESULTS.md](docs/RESULTS.md).

## Known Limitations

- **MOSAIC is not a live fact-checking system.**  
  It detects multimodal patterns that is associated with misinformation, but it can't verify claims against external knowledge sources in real time (eg. Asking for verification on a newly developing news story).

- **Out-of-context images being misused remains a difficult task.**  
  The system will preform best on paired text-image misinformation examples and may not detect cases reliably where a real image is reused in a misleading context.

- **Very short or invented phrases are harder to classify reliably.**  
  Inputs that fall far outside the training distribution, such as extremely short or vague text, may not have enough information for confident multimodal analysis.

- **Chinese support is still limited.**  
  Chinese routing into Weibo experts was verified, but end-to-end Chinese evaluation showed class imbalance, so Chinese performance is not included in the main validated benchmark claims.

- **Some websites block scraping.**  
  The `/scrape` feature depends on external site behavior, and certain domains may reject automated article extraction or image retrieval.

- **`Uncertain` is an intentional output.**  
  When expert votes are weak or conflicting, MOSAIC abstains rather than forcing a confident but unreliable prediction. This is a safety feature, not a failure.

- **Hardware requirements can be high for the full ensemble.** <br>
  The original deployment used multiple available CUDA devices on the university server. If you are running the full system on fewer GPUs or CPU-only hardware, it will require configuration changes and reduced inference speed.

## Team
<!-- Everyone: names and roles -->
| Name | Role |
|------|------|
| Amer Odobasic | Ensemble architecture, model integration, benchmarking & evaluation, backend API, project lead |
| Bhabin Pradhan | Model Training, User Documentation, Research Communication, Frontend support|
| Darren Vo | Frontend Development (HTML/CSS/JS) & API Integration, Benchmark Evaluation & Results Documentation |
| Marc Deras | URL scraper(news articles, Reddit, Twitter/X), Model training, API reference documentation |

## References

Model families and datasets used in this project are based on prior published work.
Full references are listed in [`docs/references.md`](docs/references.md).
