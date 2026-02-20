import os
import io
import sqlite3
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

# -----------------------------
# TFLite interpreter import (robust)
# -----------------------------
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception:
    # fallback to tensorflow
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "model.tflite")
LABELS_PATH = os.path.join(APP_DIR, "model", "labels.txt")
DB_PATH = os.path.join(APP_DIR, "results.db")

# If labels.txt not present, fallback to these:
DEFAULT_LABELS = ["cat", "dog"]

app = Flask(__name__)


# -----------------------------
# DB helpers
# -----------------------------
def init_db():
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
        conn.commit()


def insert_result(label: str, score: float):
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (created_at, label, score) VALUES (?, ?, ?)",
            (created_at, label, float(score)),
        )
        conn.commit()


def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        if labels:
            return labels
    return DEFAULT_LABELS


# -----------------------------
# Model helpers
# -----------------------------
def load_interpreter():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


INTERPRETER = None
LABELS = load_labels()


def get_model_io_details(interpreter: Interpreter):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # input shape often: [1, height, width, channels]
    in_shape = input_details["shape"]
    in_dtype = input_details["dtype"]

    out_shape = output_details["shape"]
    out_dtype = output_details["dtype"]

    return input_details, output_details, in_shape, in_dtype, out_shape, out_dtype


def preprocess_image(image_bytes: bytes, interpreter: Interpreter):
    input_details, _, in_shape, in_dtype, _, _ = get_model_io_details(interpreter)

    # Expect NHWC
    _, height, width, channels = in_shape
    if channels not in (1, 3):
        raise ValueError(f"Unexpected input channels: {channels}")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)

    arr = np.array(img)

    if channels == 1:
        # convert RGB to grayscale simple average
        arr = arr.mean(axis=2, keepdims=True)

    arr = arr.astype(np.float32)

    # Most TFLite image classifiers expect [0,1] float
    # If your model expects [-1,1] or uint8, adjust here.
    if in_dtype == np.uint8:
        # If model expects uint8, keep 0-255
        arr = arr.astype(np.uint8)
    else:
        arr = arr / 255.0
        arr = arr.astype(in_dtype)

    # add batch dim
    arr = np.expand_dims(arr, axis=0)

    return arr, input_details


def postprocess_output(output: np.ndarray):
    """
    Common output patterns:
    - shape (1,2): probabilities for [cat, dog]
    - shape (1,1): probability for "dog" (binary sigmoid)
    - shape (2,) or (1,2) etc.
    """
    out = output
    out = np.array(out)

    # squeeze to 1D
    out = out.squeeze()

    if out.ndim == 0:
        # scalar prob - treat as "dog" prob by convention
        dog_prob = float(out)
        cat_prob = 1.0 - dog_prob
        probs = np.array([cat_prob, dog_prob], dtype=np.float32)
    elif out.size == 1:
        dog_prob = float(out.reshape(-1)[0])
        cat_prob = 1.0 - dog_prob
        probs = np.array([cat_prob, dog_prob], dtype=np.float32)
    else:
        # assume this already represents class scores/probs
        probs = out.astype(np.float32)
        # if not normalized, normalize
        s = float(np.sum(probs))
        if s > 0 and s <= 1.5:  # already probs-ish
            pass
        else:
            # softmax
            e = np.exp(probs - np.max(probs))
            probs = e / np.sum(e)

    # map to labels
    if len(LABELS) != probs.size:
        # fallback label names
        labels = [f"class_{i}" for i in range(probs.size)]
    else:
        labels = LABELS

    idx = int(np.argmax(probs))
    label = labels[idx]
    score = float(probs[idx])
    return label, score, probs.tolist(), labels


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    global INTERPRETER
    if INTERPRETER is None:
        INTERPRETER = load_interpreter()

    if "image" not in request.files:
        return jsonify({"error": "missing form-data field 'image'"}), 400

    f = request.files["image"]
    image_bytes = f.read()
    if not image_bytes:
        return jsonify({"error": "empty image"}), 400

    try:
        inp, input_details = preprocess_image(image_bytes, INTERPRETER)
        output_details = INTERPRETER.get_output_details()[0]

        INTERPRETER.set_tensor(input_details["index"], inp)
        INTERPRETER.invoke()

        raw_out = INTERPRETER.get_tensor(output_details["index"])
        label, score, probs, labels = postprocess_output(raw_out)

        insert_result(label, score)

        return jsonify(
            {
                "label": label,
                "score": score,
                "labels": labels,
                "probs": probs,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/results")
def results():
    # returns latest 50
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, label, score FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    init_db()
    # NOTE:
    # - Webcam requires HTTPS OR http://localhost in modern browsers.
    # - For phone testing in LAN, recommend running behind HTTPS (ngrok / local cert).
    app.run(host="0.0.0.0", port=5000, debug=True)
