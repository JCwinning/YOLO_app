# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated YOLO11 object detection application built with Streamlit that supports multiple processing backends including local YOLO models and cloud-based AI models (Qwen-Image-Edit and Gemini 2.5 Flash Image). It provides a comprehensive web interface for object detection on images, videos, and live camera feeds with full internationalization support.

## Development Commands

### Environment Setup

```bash
# Install dependencies using uv (recommended)
uv sync

# Start the Streamlit application
streamlit run app.py
```

The app will be available at http://localhost:8501

### Dependencies

The project uses `uv` as the package manager with Python 3.12+. All dependencies are defined in `pyproject.toml` including:
- Core ML libraries: `ultralytics`, `torch`, `opencv-python`
- Web framework: `streamlit`
- Cloud integrations: `dashscope`, `openai`, `requests`
- Image processing: `pillow`, `numpy`

### Development Workflow

The application automatically handles:
- Model downloading (YOLO models are pre-downloaded)
- Device detection (MPS for Apple Silicon, CPU fallback)
- Temporary file management in `detect/` directory
- Session state persistence for API keys and media

## Architecture

### Multi-Model Processing Architecture

The application supports three different processing backends:

1. **Local YOLO Models** (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt):
   - Runs locally using Ultralytics framework
   - Supports MPS acceleration on Apple Silicon
   - Processes images, videos, and camera input
   - Returns structured detection data (bounding boxes, confidence, classes)

2. **Qwen-Image-Edit** (DashScope API):
   - Cloud-based image generation with visual annotations
   - Returns annotated images with bounding boxes and labels
   - Images only (no video support)
   - Requires session-based API key configuration

3. **Gemini 2.5 Flash Image** (OpenRouter API):
   - Cloud-based model supporting both annotated image generation and text analysis
   - Uses `modalities: ["image", "text"]` for image generation
   - Falls back to text analysis if image generation fails
   - Handles both data URLs and HTTP URLs for generated images

### Session State Management

The application uses comprehensive session state management (lines 26-63 in app.py):
- Media storage: `current_image`, `current_video`, `camera_frame`, `camera_active`
- Processing flags: `qwen_processed`, `gemini_processed`
- API key storage: `dashscope_api_key`, `openrouter_api_key` (session-only)
- Language preference: `language`, `input_method_index`

### Internationalization System

The application supports bilingual interface (English/Chinese):
- **Translation file**: `language.py` contains 113+ translatable strings
- **Translation function**: `get_translation(key, **kwargs)` with parameter formatting
- **Language switcher**: Button in top-right corner toggles between EN/中文
- **Session persistence**: Language preference maintained during session

### Video Processing Pipeline

Video processing uses subprocess execution to capture real-time progress:
- **Progress parsing**: Regex pattern matching for frame progress
- **Output directory**: `detect/` with timestamp-based file naming
- **Temporary management**: Automatic cleanup in finally blocks

### Image Processing for Cloud APIs

Cloud API integration requires image compression:
- **Size limits**: Progressive compression to 8MB limit
- **Format support**: JPEG and WEBP with quality optimization
- **Encoding**: Base64 with MIME type headers
- **Background handling**: White background conversion for JPEG transparency

## Key Technical Patterns

### API Integration Patterns

**Qwen-Image-Edit Integration** (lines 386-485):
- Session-based API key management
- Multi-format image compression with fallback strategies
- Base64 encoding with MIME type specification
- URL-based result image downloading

**Gemini 2.5 Flash Image Integration** (lines 487-625):
- Direct HTTP requests to OpenRouter API
- Dual modality support with fallback text analysis
- Error handling for different response types
- Session-only API key storage (no keyring usage)

### Error Handling Strategy

- Graceful degradation between cloud models
- Comprehensive validation for file types and API responses
- User-friendly error messages with specific guidance
- Temporary file cleanup in finally blocks

### Device Detection and Acceleration

Automatic device detection with fallback:
- **MPS (Metal Performance Shaders)**: Apple Silicon acceleration
- **CPU**: Fallback for other platforms
- **Cloud APIs**: Remote processing for Qwen and Gemini models

## File Structure

```
/Users/jinchaoduan/Documents/post_project/YOLO_app/
├── app.py                    # Main application (multi-model, i18n, full feature set)
├── app backup.py            # Backup version (simpler, no i18n/cloud models)
├── language.py              # Translation system (en/zh, 113+ strings)
├── pyproject.toml           # Dependencies and project configuration
├── uv.lock                  # Lock file for reproducible builds
├── README.md                # User-facing documentation
├── CLAUDE.md                # This development guide
├── ui.md                    # UI design specifications
├── .python-version          # Python version specification (3.12)
├── .gitignore               # Git ignore rules
├── yolo11x.pt               # Pre-downloaded YOLO11x model (114MB)
├── yolo11s.pt               # Pre-downloaded YOLO11s model (19MB)
└── detect/                  # Runtime directory for video outputs
```

## Important Implementation Details

### Model Selection Logic

Model selection affects processing behavior:
- **Local YOLO models**: Support all input types (images, videos, camera)
- **Qwen-Image-Edit**: Images only, requires DashScope API key
- **Gemini 2.5 Flash Image**: Images only, requires OpenRouter API key
- **Default model**: yolo11x.pt (highest accuracy)

### Input Method State Management

The application preserves input method selection across language switches:
- **Storage**: Session state stores index (0=Upload, 1=URL, 2=Camera)
- **Preservation**: Language changes maintain selected input method
- **UI Consistency**: Radio button index persists during rerun

### Image Hiding After Detection

After successful detection, the input image section is automatically hidden:
- **Detection check**: Monitors `results`, `qwen_processed`, `gemini_processed`
- **Conditional display**: Shows input image only when no results exist
- **Reset behavior**: Input section reappears on new uploads or rotation

### API Key Management

Both cloud models use session-only API key storage:
- **No persistence**: Keys lost on browser refresh
- **Sidebar configuration**: Input fields appear when cloud models selected
- **Security**: No keyring or file-based storage
- **Validation**: Key presence checked before API calls