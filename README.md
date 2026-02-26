# Cat vs Dog Image Classifier

A simple cat vs dog image classifier using TensorFlow Lite. Upload images or use your webcam to get instant predictions!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17+-orange.svg)
![TFLite](https://img.shields.io/badge/TFLite-Model-green.svg)

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

2. **Set up Python virtual environment**

   **Option A: Activate existing environment**
   ```bash
   # Windows
   catdog_webcam\Scripts\activate

   # Linux/Mac
   source catdog_webcam/bin/activate
   ```

   **Option B: Create new virtual environment**
   ```bash
   # Create venv
   python -m venv venv

   # Activate it
   # Windows:
   venv\Scripts\activate

   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**

   For Flask:
   ```bash
   pip install -r requirements-flask.txt
   ```

   For Streamlit:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app** (see Quick Start above)

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
| `http://localhost:8501` | Yes | No |
| `http://192.168.x.x:8501` | Yes | No (HTTP) |
| Deploy to Streamlit Cloud | Yes | Yes (HTTPS) |

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

## Official Documentation

### Core Technologies

**TensorFlow & TensorFlow Lite**
- [TensorFlow Official Docs](https://www.tensorflow.org/api_docs)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [TensorFlow Lite Python API](https://www.tensorflow.org/lite/api_docs/python/tf/lite)
- [TFLite Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization)

**Python Libraries**
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Pillow (PIL) Documentation](https://pillow.readthedocs.io/en/stable/)
- [SQLite3 Python Module](https://docs.python.org/3/library/sqlite3.html)

### Web Frameworks

**Streamlit**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit API Reference](https://docs.streamlit.io/library/api-reference)
- [Streamlit Camera Input](https://docs.streamlit.io/library/api-reference/widgets/st.camera_input)
- [Streamlit Cloud Deployment](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

**Flask**
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask Quickstart](https://flask.palletsprojects.com/en/3.0.x/quickstart/)
- [Flask Request Object](https://flask.palletsprojects.com/en/3.0.x/api/#flask.Request)
- [Flask File Uploads](https://flask.palletsprojects.com/en/3.0.x/patterns/fileuploads/)

### Browser APIs (for Flask implementation)

**WebRTC & Camera Access**
- [MediaDevices.getUserMedia()](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- [MediaStream API](https://developer.mozilla.org/en-US/docs/Web/API/MediaStream)
- [WebRTC API Overview](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)

**Canvas & Image Processing**
- [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
- [HTMLCanvasElement.toBlob()](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/toBlob)
- [HTMLCanvasElement.toDataURL()](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/toDataURL)

**HTTP & Fetch**
- [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
- [FormData](https://developer.mozilla.org/en-US/docs/Web/API/FormData)
- [Blob](https://developer.mozilla.org/en-US/docs/Web/API/Blob)

### Deployment

**Streamlit Cloud**
- [Streamlit Cloud Get Started](https://streamlit.io/cloud)
- [App Dependencies](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)

### Security & HTTPS

**Camera Access Requirements**
- [Secure Contexts (HTTPS)](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts)
- [Permissions API](https://developer.mozilla.org/en-US/docs/Web/API/Permissions_API)
- [Feature Policy: camera](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Feature-Policy/camera)

### Additional Resources

**Python Version Management**
- [pyenv Documentation](https://github.com/pyenv/pyenv)
- [Python Version Specification](https://docs.python.org/3/using/index.html)

**Git & Version Control**
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)

**Machine Learning Concepts**
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Image Classification Guide](https://www.tensorflow.org/tutorials/images/classification)
