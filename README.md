# Cat vs Dog Image Classifier

A simple cat vs dog image classifier using TensorFlow Lite. Upload images or use your webcam to get instant predictions!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TFLite](https://img.shields.io/badge/TFLite-2.14.0-orange.svg)

## Features

- Upload images for classification
- Webcam capture (works on mobile with HTTPS)
- Fast TFLite model (~2.3MB)
- Prediction history tracking
- Two UI options: Streamlit (simple) or Flask (customizable)

## Quick Start

### Option 1: Flask

Traditional web app with custom HTML/CSS/JavaScript.

```bash
# Install
pip install -r requirements-flask.txt

# Run
python app.py

# Opens at http://localhost:5000
```

### Option 2: Streamlit

Simple Python-only interface, easiest for deployment.

```bash
# Install
pip install -r requirements.txt

# Run
streamlit run streamlit_app.py

# Opens at http://localhost:8501
```

## Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/liuxhy/catdog_webcam.git
   cd catdog_webcam
   ```

2. **Install dependencies**

   For Flask:
   ```bash
   pip install -r requirements-flask.txt
   ```

   For Streamlit:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app** (see Quick Start above)

## Project Structure

```
catdog_webcam/
├── app.py                  # Flask app (custom UI)
├── streamlit_app.py        # Streamlit app (simple UI)
├── requirements-flask.txt  # Flask dependencies
├── requirements.txt        # Streamlit dependencies (for cloud deployment)
├── model/
│   ├── model.tflite       # TFLite model (2.3MB)
│   └── labels.txt         # Class labels: cat, dog
├── templates/
│   └── index.html         # Flask UI
└── results.db             # SQLite database (auto-created)
```

## Usage

### Flask App

1. Run: `python app.py`
2. Open browser to `http://localhost:5000`
3. Use "Upload" section or "Webcam" section
4. Get instant classification results

### Streamlit App

1. Run: `streamlit run streamlit_app.py`
2. Choose "Upload Image" or "Camera" tab
3. Upload/capture an image of a cat or dog
4. Click "Classify" to see results

## Camera Access

**Important:** Webcam requires HTTPS or localhost for security.

| Access Method | Works? | Camera on Mobile? |
|---------------|--------|-------------------|
| `http://localhost:8501` | ✅ Yes | ❌ No |
| `http://192.168.x.x:8501` | ✅ Yes | ❌ No (HTTP) |
| Deploy to Streamlit Cloud | ✅ Yes | ✅ Yes (HTTPS) |

**For mobile camera access:** Deploy to Streamlit Cloud (automatic HTTPS).

## Model Information

- **Format:** TensorFlow Lite (.tflite)
- **Size:** ~2.3MB
- **Input:** RGB images (auto-resized)
- **Output:** Binary classification (cat vs dog)
- **Classes:** cat, dog

## API Endpoints (Flask)

### `POST /predict`
Classify an image.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `image` (file)

**Response:**
```json
{
  "label": "cat",
  "score": 0.92,
  "labels": ["cat", "dog"],
  "probs": [0.92, 0.08]
}
```

### `GET /results`
Get recent predictions.

**Response:**
```json
[
  {
    "id": 1,
    "created_at": "2024-02-26T12:34:56",
    "label": "dog",
    "score": 0.87
  }
]
```

## Troubleshooting

### Camera not working

**Problem:** No camera permission prompt.

**Solution:**
- Access via `http://localhost` (not IP address)
- Or deploy to Streamlit Cloud for HTTPS

### Model not found

**Problem:** `Model not found: model/model.tflite`

**Solution:**
- Ensure `model/model.tflite` exists
- Check file permissions

### Import error

**Problem:** `No module named 'tflite_runtime'`

**Solution:**
```bash
pip install tflite-runtime
# Or use full TensorFlow:
pip install tensorflow
```

## Dependencies

**Flask (requirements-flask.txt):**
- flask
- pillow >= 10.0.0
- numpy >= 1.23.0
- tensorflow >= 2.17.0
- gunicorn

**Streamlit (requirements.txt):**
- streamlit >= 1.12.0
- pillow >= 10.0.0
- numpy >= 1.23.0
- tensorflow >= 2.17.0

## License

MIT License

## Contributing

Pull requests welcome!

## Acknowledgments

- TensorFlow Lite for efficient inference
- Streamlit for easy UI
- Flask for web framework
