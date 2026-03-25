# YOLO11 Object Detection Streamlit App

## Project Overview
This project is a comprehensive **Streamlit web application** for object detection using the **YOLO11** architecture (via `ultralytics`). It supports real-time object detection on images, videos, and live camera feeds. Additionally, it integrates cloud-based AI models (**Qwen-Image-Edit** via DashScope and **Gemini 3 Pro Image** via OpenRouter) for generating annotated images with bounding boxes.

**Key Features:**
*   **Multi-Model Support:** Runs local YOLO11 models (n, s, m, l, x) and cloud-based models.
*   **Input Flexibility:** Supports file uploads (images/videos), URL inputs, and front camera capture.
*   **Bilingual Interface:** Full English and Chinese (中文) support, toggled via UI.
*   **Smart UI:** Dynamic interface that manages state and hides inputs after processing for cleaner results.
*   **Hardware Acceleration:** Automatically detects and uses MPS (Apple Silicon) or CUDA (NVIDIA) if available.

## Tech Stack
*   **Frontend/UI:** [Streamlit](https://streamlit.io/)
*   **Core Detection:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
*   **Image Processing:** OpenCV (`cv2`), Pillow (`PIL`), NumPy
*   **Cloud Integrations:** DashScope (Qwen), OpenRouter (Gemini)
*   **Language:** Python 3.12+

## Setup & Usage

### 1. Prerequisites
*   Python 3.12 or higher.
*   `uv` package manager (recommended) or `pip`.

### 2. Installation
**Using uv (Recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### 3. Running the App
Execute the following command in the project root:
```bash
streamlit run app.py
```
This will launch the application in your default web browser (typically at `http://localhost:8501`).

## Key Files & Structure

*   **`app.py`**: The main entry point. Handles the Streamlit UI layout, session state management, input processing, and model inference logic.
*   **`language.py`**: Contains the dictionary-based translation system (`translations`) and helper functions for bilingual support.
*   **`pyproject.toml`**: Project metadata and dependency definitions (PEP 621 compliant).
*   **`requirements.txt`**: Standard pip requirements file.
*   **`detect/`**: Directory where processed video outputs are saved.
*   **`images/`**: Directory containing sample images or assets used in documentation.

## Development Conventions

*   **Localization:** All UI text should be retrieved via the `get_translation(key)` function in `app.py`, which looks up keys in `language.py`. When adding new UI elements, ensure corresponding English and Chinese keys are added to `language.py`.
*   **Session State:** The app relies heavily on `st.session_state` to persist data (images, results, API keys) across re-runs. Always check/initialize session state variables before use.
*   **Model Integration:** New models should be added to the model selection list in `app.py` and handled in the conditional logic for inference. Cloud models require specific API key handling.
*   **Code Style:** Follow standard Python PEP 8 guidelines.
