"""
Streamlit app for Cat vs Dog classification using TFLite.
Simple interface with upload and camera support.
"""
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from PIL import Image

# TFLite interpreter import
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter


# Configuration
MODEL_PATH = "model/model.tflite"
LABELS_PATH = "model/labels.txt"
DB_PATH = "results.db"
DEFAULT_LABELS = ["cat", "dog"]

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶", layout="centered")


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
@st.cache_resource
def load_labels():
    """Load class labels."""
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f if line.strip()]
        if labels:
            return labels
    return DEFAULT_LABELS


@st.cache_resource
def load_model():
    """Load TFLite model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def predict_image(image: Image.Image):
    """Run inference on image."""
    interpreter = load_model()
    labels = load_labels()

    # Preprocess
    input_details = interpreter.get_input_details()[0]
    _, height, width, channels = input_details["shape"]

    img = image.convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)

    if input_details["dtype"] == np.uint8:
        arr = arr.astype(np.uint8)
    else:
        arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    # Inference
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])

    # Results
    probs = output.squeeze().astype(np.float32)
    idx = int(np.argmax(probs))
    label = labels[idx]
    score = float(probs[idx])

    return label, score, probs, labels


# Main app
def main():
    init_db()

    st.title("🐱🐶 Cat vs Dog Classifier")
    st.write("Upload an image or use your camera to classify cats and dogs!")

    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Classify", key="upload"):
                with st.spinner("Classifying..."):
                    label, score, probs, labels = predict_image(image)
                    save_prediction(label, score)

                    emoji = "🐱" if label == "cat" else "🐶"
                    st.success(f"## {emoji} {label.upper()} ({score:.1%})")

                    for lbl, prob in zip(labels, probs):
                        st.write(f"{lbl}: {prob:.1%}")

    with tab2:
        camera_image = st.camera_input("Take a picture")

        if camera_image:
            image = Image.open(camera_image)

            if st.button("Classify", key="camera"):
                with st.spinner("Classifying..."):
                    label, score, probs, labels = predict_image(image)
                    save_prediction(label, score)

                    emoji = "🐱" if label == "cat" else "🐶"
                    st.success(f"## {emoji} {label.upper()} ({score:.1%})")

                    for lbl, prob in zip(labels, probs):
                        st.write(f"{lbl}: {prob:.1%}")


if __name__ == "__main__":
    main()
