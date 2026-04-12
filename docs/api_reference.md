# MOSAIC API Reference
 
> Detailed technical specifications for the MOSAIC Flask backend, including request formats, response schemas, and system thresholds.
 
The MOSAIC backend is built using Flask and serves as the bridge between the frontend interfaces and the ensemble inference engine. All communication is handled via JSON unless otherwise specified.
 
## Base URLs
 
During development and deployment, use the following base addresses:
 
- **Local Development:** `http://127.0.0.1:5000`
- **Public Demo:** Set by your specific ngrok URL (e.g., `https://your-domain.ngrok-free.app`)
 
---
 
## 1. GET `/health`
 
Checks the status of the Flask backend and verifies that the ensemble of 28 expert models is fully loaded and ready for inference.
 
**Request:** `GET /health`
 
**Response Example:**
 
    {
      "status": "ok",
      "models_loaded": true,
      "expert_count": 28
    }
 
---
 
## 2. POST `/predict`
 
Submit a text caption and an image file for multimodal classification through the full ensemble.
 
**Request Type:** `multipart/form-data`
 
| Field | Type | Required | Description | 
| :--- | :--- | :--- | :--- | 
| `text` | string | Yes | The caption or post text to analyze | 
| `image` | file | Yes | The image file (JPG, PNG, WEBP) | 
 
**Example Curl:**
 
    curl -X POST http://127.0.0.1:5000/predict \
      -F "text=Shark seen swimming down I-10 in Lake Charles LA" \
      -F "image=@/path/to/image.jpg"
 
**Response Example:**
 
    {
      "verdict": "Fake",
      "confidence": 0.79,
      "vote_strength": 0.56,
      "agreement": 0.78,
      "language": "EN",
      "image_used": true,
      "why_fake": "Decision driven primarily by MCAN (snopes), MoPeD (snopes)...",
      "supporting_experts": ["MCAN (snopes)", "MoPeD (snopes)", "COOLANT (snopes)"],
      "expert_breakdown": {
        "MoPeD (snopes)": {"verdict": "Fake", "real": 0.04, "fake": 0.96, "weight": 1.0},
        "COOLANT (snopes)": {"verdict": "Fake", "real": 0.08, "fake": 0.92, "weight": 1.0}
      }
    }
 
---
 
## 3. POST `/scrape`
 
Submit a URL to automatically extract content and classify it in one step. The backend utilizes `scraper.py` to fetch remote assets before routing them to the inference engine.
 
**Request Type:** `application/json`
 
| Field | Type | Required | Description | 
| :--- | :--- | :--- | :--- | 
| `url` | string | Yes | The full URL of the post or news article | 
 
**Example Curl:**
 
    curl -X POST http://127.0.0.1:5000/scrape \
      -H "Content-Type: application/json" \
      -d '{"url": "https://www.snopes.com/fact-check/example/"}'
 
**Response Example:**
 
    {
      "verdict": "Uncertain",
      "confidence": 0.54,
      "vote_strength": 0.24,
      "agreement": 0.69,
      "language": "EN",
      "image_used": true,
      "warning": "This appears to be a live update page. Results may be less reliable.",
      "why_fake": null,
      "supporting_experts": [],
      "expert_breakdown": {
        "MoPeD (snopes)": {"verdict": "Real", "real": 0.80, "fake": 0.20, "weight": 1.0},
        "COOLANT (snopes)": {"verdict": "Fake", "real": 0.30, "fake": 0.70, "weight": 1.0}
      }
    }
 
> **Note:** The `warning` field only appears in `/scrape` responses, not `/predict`. It is present when the scraper detects a potentially unreliable source (e.g. live update pages, missing images). It will be `null` or absent when scraping succeeds cleanly.
 
---
 
## Scraping Support & Limitations
 
The system uses Playwright and `newspaper3k` for automated extraction. Performance varies based on platform anti-scraping measures.
 
| Platform | Support | Methodology | Notes | 
| :--- | :--- | :--- | :--- | 
| **News Articles** | Full | Playwright/newspaper3k | Works on most major news sites | 
| **Reddit** | Full | Public .json API | No API key required | 
| **X / Twittere** | Full | Nitter | May fail if Nitter instance is down | 
| **Facebook** | None | N/A | Blocked by Meta; use manual upload | 
| **Instagram** | None | N/A | Blocked by Meta; use manual upload | 
| **Live Blogs** | Partial | Playwright | Returns warning; text may be fragmented | 
 
---
 
## Error Responses
 
Errors return a JSON object with an `error` field and an appropriate HTTP status code.
 
| Status | Meaning | Typical Cause | 
| :--- | :--- | :--- | 
| **400** | Bad Request | Missing text, image, or URL fields | 
| **422** | Unprocessable Entity | Scraping failed or platform is unsupported | 
| **500** | Internal Error | Inference failure or GPU memory issues | 
 
**Example Error Response:**
 
    {
      "error": "Instagram cannot be scraped. Please copy and paste the post text manually."
    }
 
---
 
## Ensemble Logic & Thresholds
 
To prevent "hallucinating" a verdict on ambiguous data, MOSAIC utilizes strict confidence gates within `model_manager.py`:
 
- **Uncertainty Trigger:** The system returns `Uncertain` if:
  - `vote_strength < 0.20`
  - OR `agreement < 0.65`
- **Weighting:** Final results are a product of `domain_weight` (how relevant the model is to the input domain) and `reliability_weight` (performance metrics from benchmarking).
- **Language Routing:** The system automatically detects input language. Chinese (ZH) inputs activate the Weibo-trained expert family.