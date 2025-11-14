# Yolo object detection streamlit_app.py

Build a Streamlit app named streamlit_app.py that performs object detection using YOLO11l (from the Ultralytics YOLO11
 framework).

🎯 Goal

Create a user-friendly Streamlit web interface to:

Accept an image (upload or URL).

Run YOLO11l detection using Apple Silicon (MPS) or CPU.

Display the annotated output image with bounding boxes, labels, and confidence scores.

Show a summary table of detected objects and allow image download.

⚙️ Dependencies

Add to requirements.txt:

ultralytics>=8.3.0
streamlit
pillow
opencv-python
numpy
torch>=2.2


Make sure PyTorch is installed with MPS support for Apple Silicon:

pip install torch torchvision torchaudio

🧩 Architecture Overview
/yolo11_app
  ├─ streamlit_app.py
  ├─ requirements.txt
  ├─ utils.py                # helper functions (optional)
  ├─ models/                 # optional local .pt models
  └─ README.md

🖥️ UI Design (Streamlit)
Top Section

Title: "YOLO11 Object Detection (Ultralytics)"

Short description about YOLO11 and Apple Silicon support.

Sidebar

Image Input:

Radio: “📸 Upload Image” | “🌐 URL Input”

File uploader (accepts .jpg, .jpeg, .png)

Text input for URL (if selected)

Model Selector:

Dropdown: ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"]

Default: yolo11l.pt

Tooltip: Use smaller models if performance is slow.

Confidence threshold: slider (0.1 → 0.9, default 0.25)

Device:

Auto-detect (“MPS” if available, else “CPU”)

Button: “🚀 Start Detection”


