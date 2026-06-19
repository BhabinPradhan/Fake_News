"""
api.py
------
Flask backend for MOSAIC — Multimodal Fake News Detection System.
Exposes two endpoints:
  POST /predict  — runs the ensemble on text + image
  POST /scrape   — scrapes text and image from a URL

Run with:
    python api.py
Or for production:
    gunicorn -w 1 -b 0.0.0.0:5000 api:app
"""

import os
import sys
import tempfile
import base64
from io import BytesIO

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "MoPeD"))

app = Flask(__name__)
CORS(app)  # This will be restricted later in production to only allow requests from the frontend domain

# Load ModelManager once at startup
# All 20+ models load into GPU memory here and  takes ~2 minutes but only happens once per server session
print("Loading ModelManager — this may take a few minutes...")
try:
    from model_manager import ModelManager
    manager = ModelManager()
    print("ModelManager ready.")
except Exception as e:
    print(f"FATAL: ModelManager failed to load: {e}")
    manager = None


# Convert the PIL image to a base64 string for JSON serialization
def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Sanitize the result dict for JSON serialization 
def sanitize_result(result):
    for score in result.get("all_scores", []):
        score["Real"]   = round(float(score["Real"]),   3)
        score["Fake"]   = round(float(score["Fake"]),   3)
        score["weight"] = round(float(score["weight"]), 3)
        score["margin"] = round(float(score["margin"]), 3)
    return result

# Basic route to serve the frontend (the index.html) if it is needed
@app.route("/")
def index():
    return render_template("index.html")

# POST /predict
# - text  (string)  : the post text or headline
# - image (file)    : the image file (JPG, PNG, WebP)
# It will then return a JSON with verdict, confidence, per-expert scores, and its metadata.
@app.route("/predict", methods=["POST"])
def predict():
    if manager is None:
        return jsonify({"error": "ModelManager failed to initialize on startup"}), 503

    # Validate inputs 
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400

    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "Image is required — both text and image are needed"}), 400

    # Check it's actually an image
    if not image_file.content_type.startswith("image/"):
        return jsonify({"error": f"File must be an image, got: {image_file.content_type}"}), 400

    # Run the inference
    tmp_path = None
    try:
        # Save uploaded image to a temp file. ModelManager expects a file path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        Image.open(image_file).convert("RGB").save(tmp.name)
        tmp_path = tmp.name
        tmp.close()

        result = manager.get_prediction(text, tmp_path)
        return jsonify(sanitize_result(result))

    except ValueError as e:
        # Input validation errors from ModelManager (empty text, missing image)
        return jsonify({"error": str(e)}), 400

    except RuntimeError as e:
        # Voting errors. No active experts or weight issues
        return jsonify({"error": f"Ensemble error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    finally:
        # Clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# POST /scrape 
# Expects JSON with:
#   - url (string) : the URL to scrape
# Returns JSON with:
#   - text  (string)      : extracted article text
#   - image (string|null) : base64 encoded image, or null if none found
    
@app.route("/scrape", methods=["POST"])
def scrape():
    data = request.get_json(silent=True)
    if not data or not data.get("url"):
        return jsonify({"error": "JSON body with 'url' field is required"}), 400

    url = data["url"].strip()
    if not url.startswith("http"):
        return jsonify({"error": "URL must start with http or https"}), 400

    try:
        from scraper import get_scraped_data
        scraped = get_scraped_data(url)

        image_b64 = None
        if scraped.get("image"):
            try:
                image_b64 = pil_to_base64(scraped["image"])
            except Exception:
                image_b64 = None  # Image conversion failed, so return text only

        return jsonify({
            "text":  scraped.get("text", ""),
            "image": image_b64,
            "warning": scraped.get("warning"),
        })

    except Exception as e:
        return jsonify({"error": f"Failed to scrape: {str(e)}"}), 500

# GET /health
# A simple health check endpoint to verify the server is running and ModelManager loaded successfully.
@app.route("/health", methods=["GET"])
def health():
    """
    Simple health check — useful for verifying the server is running.
    Returns 200 if ModelManager loaded successfully, 503 if not.
    """
    if manager is None:
        return jsonify({"status": "error", "message": "ModelManager not loaded"}), 503
    return jsonify({"status": "ok", "message": "MOSAIC backend ready"}), 200

# The Entry point 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
