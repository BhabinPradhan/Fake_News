import os
import sys
import tempfile
import base64
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "MoPeD"))

app = Flask(__name__)
CORS(app)

# Load the ensemble once when the server starts.
# This takes a while because all expert models are loaded up front.
print("Loading ModelManager — this may take a few minutes...")
try:
    from model_manager import ModelManager

    manager = ModelManager()
    print("ModelManager ready.")
except Exception as e:
    print(f"FATAL: ModelManager failed to load: {e}")
    manager = None


# Convert a PIL image into a base64 string for JSON responses.
def pil_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Round the score fields so the JSON response is easier to read.
def sanitize_result(result):
    for score in result.get("all_scores", []):
        score["Real"] = round(float(score["Real"]), 3)
        score["Fake"] = round(float(score["Fake"]), 3)
        score["weight"] = round(float(score["weight"]), 3)
        score["margin"] = round(float(score["margin"]), 3)
    return result


# Run the ensemble on text and image input.
@app.route("/predict", methods=["POST"])
def predict():
    if manager is None:
        return jsonify({"error": "ModelManager failed to initialize on startup"}), 503

    # Check the required inputs first.
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400

    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "Image is required — both text and image are needed"}), 400

    # Make sure the uploaded file is an image.
    if not image_file.content_type.startswith("image/"):
        return jsonify({"error": f"File must be an image, got: {image_file.content_type}"}), 400

    tmp_path = None
    try:
        # Save the image to a temp file because ModelManager expects a path.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        Image.open(image_file).convert("RGB").save(tmp.name)
        tmp_path = tmp.name
        tmp.close()

        result = manager.get_prediction(text, tmp_path)
        return jsonify(sanitize_result(result))

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except RuntimeError as e:
        return jsonify({"error": f"Ensemble error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    finally:
        # Remove the temp image after the request finishes.
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Scrape article text and image from a URL.
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
                image_b64 = None

        return jsonify({
            "text": scraped.get("text", ""),
            "image": image_b64,
            "warning": scraped.get("warning"),
        })

    except Exception as e:
        return jsonify({"error": f"Failed to scrape: {str(e)}"}), 500


# Simple health check for the backend.
@app.route("/health", methods=["GET"])
def health():
    if manager is None:
        return jsonify({"status": "error", "message": "ModelManager not loaded"}), 503
    return jsonify({"status": "ok", "message": "MOSAIC backend ready"}), 200


# Start the Flask server.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
