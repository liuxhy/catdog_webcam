"""
Flask web app for Cat vs Dog classification using TFLite.
Supports file upload and webcam capture.
"""
import os
import io
import sqlite3
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

# TFLite interpreter import
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter


# Configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "model.tflite")
LABELS_PATH = os.path.join(APP_DIR, "model", "labels.txt")
DB_PATH = os.path.join(APP_DIR, "results.db")
DEFAULT_LABELS = ["cat", "dog"]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}

app = Flask(__name__)
INTERPRETER = None
LABELS = []


# Database functions
def init_db():
    """Initialize database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                label TEXT NOT NULL,
                score REAL NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON predictions (created_at DESC)"
        )
        conn.commit()


def save_prediction(label: str, score: float):
    """Save prediction to database."""
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, label, score) VALUES (?, ?, ?)",
            (created_at, label, float(score)),
        )
        conn.commit()


# Model functions
def load_labels():
    """Load class labels."""
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f if line.strip()]
        if labels:
            return labels
    return DEFAULT_LABELS


def load_model():
    """Load TFLite model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes: bytes):
    """Preprocess image for model input."""
    input_details = INTERPRETER.get_input_details()[0]
    _, height, width, channels = input_details["shape"]

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)

    if input_details["dtype"] == np.uint8:
        arr = arr.astype(np.uint8)
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0), input_details


def postprocess_output(output: np.ndarray):
    """Process model output."""
    probs = output.squeeze().astype(np.float32)
    idx = int(np.argmax(probs))
    label = LABELS[idx]
    score = float(probs[idx])
    return label, score, probs.tolist()


# Routes
@app.get("/")
def index():
    """Render main page."""
    return render_template("index.html")


@app.post("/predict")
def predict():
    """Classify uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    # Validate file type
    if file.filename and not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty image"}), 400

    # Validate file size (max 10MB)
    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "File too large (max 10MB)"}), 400

    try:
        inp, input_details = preprocess_image(image_bytes)
        output_details = INTERPRETER.get_output_details()[0]

        INTERPRETER.set_tensor(input_details["index"], inp)
        INTERPRETER.invoke()

        raw_out = INTERPRETER.get_tensor(output_details["index"])
        label, score, probs = postprocess_output(raw_out)

        save_prediction(label, score)

        return jsonify({
            "label": label,
            "score": score,
            "labels": LABELS,
            "probs": probs,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/results")
def results():
    """Get recent predictions."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, label, score FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    init_db()

    # Load model at startup
    try:
        INTERPRETER = load_model()
        LABELS = load_labels()
        print(f"✓ Model loaded: {MODEL_PATH}")
        print(f"✓ Labels: {LABELS}")
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)

    print("\n Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)