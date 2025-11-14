import streamlit as st
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os
import re
import sys
import subprocess
import cv2
import time
import base64
from language import translations

# Initialize language session state
if "language" not in st.session_state:
    st.session_state["language"] = "en"

# Initialize input method session state (store as index 0, 1, 2)
if "input_method_index" not in st.session_state:
    st.session_state["input_method_index"] = 0  # Default to first option (Upload)

def get_translation(key, **kwargs):
    """Translation function that uses the current session language"""
    lang = st.session_state.get("language", "en")
    text = translations[lang].get(key, translations["en"].get(key, key))
    return text.format(**kwargs) if kwargs else text

try:
    from dashscope import MultiModalConversation

    dashscope_available = True
except ImportError:
    dashscope_available = False
    MultiModalConversation = None

if "current_image" not in st.session_state:
    st.session_state["current_image"] = None

if "uploaded_image_bytes" not in st.session_state:
    st.session_state["uploaded_image_bytes"] = None

if "current_video" not in st.session_state:
    st.session_state["current_video"] = None

if "uploaded_video_bytes" not in st.session_state:
    st.session_state["uploaded_video_bytes"] = None

if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False

if "camera_frame" not in st.session_state:
    st.session_state["camera_frame"] = None

if "qwen_processed" not in st.session_state:
    st.session_state["qwen_processed"] = False

if "gemini_processed" not in st.session_state:
    st.session_state["gemini_processed"] = False

# Set DashScope API URL if available
if dashscope_available:
    import dashscope

    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1/"


def parse_yolo_progress(line):
    """Parse YOLO progress output to extract frame information."""
    # Pattern to match: video 1/1 (frame 258/267) or similar
    pattern = r"video \d+/\d+ \(frame (\d+)/(\d+)\)"
    match = re.search(pattern, line)
    if match:
        current_frame = int(match.group(1))
        total_frames = int(match.group(2))
        progress = (current_frame / total_frames) * 100
        return current_frame, total_frames, progress
    return None, None, None


def encode_image_to_base64(image):
    """Encode PIL Image to base64 string for DashScope API with size compression."""
    max_size_bytes = 8 * 1024 * 1024  # 8MB limit (under 10MB API limit)

    # Try different compression levels and formats
    formats_and_qualities = [
        ("JPEG", 95),
        ("JPEG", 85),
        ("JPEG", 75),
        ("JPEG", 65),
        ("WEBP", 95),
        ("WEBP", 85),
        ("WEBP", 75),
        ("WEBP", 65),
    ]

    for fmt, quality in formats_and_qualities:
        buffer = BytesIO()

        # Convert RGBA to RGB for JPEG if needed
        if fmt == "JPEG" and image.mode in ("RGBA", "LA"):
            # Create a white background for transparent images
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "RGBA":
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            image_to_save = background
        else:
            image_to_save = image

        # Save with compression
        if fmt == "WEBP":
            image_to_save.save(buffer, format=fmt, quality=quality, method=6)
        else:
            image_to_save.save(buffer, format=fmt, quality=quality, optimize=True)

        image_bytes = buffer.getvalue()

        # Check if size is acceptable
        if len(image_bytes) <= max_size_bytes:
            mime_type = f"image/{fmt.lower()}"
            encoded_string = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:{mime_type};base64,{encoded_string}"

    # If still too large, try resizing
    max_dimension = 1024  # Max width/height
    img_copy = image.copy()

    while max(img_copy.size) > max_dimension:
        # Reduce size by 20%
        new_size = tuple(int(dim * 0.8) for dim in img_copy.size)
        img_copy = img_copy.resize(new_size, Image.Resampling.LANCZOS)

        # Try compression again
        for fmt, quality in formats_and_qualities[:4]:  # Try JPEG first for resizing
            buffer = BytesIO()
            if fmt == "JPEG" and img_copy.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img_copy.size, (255, 255, 255))
                if img_copy.mode == "RGBA":
                    background.paste(img_copy, mask=img_copy.split()[-1])
                else:
                    background.paste(img_copy)
                image_to_save = background
            else:
                image_to_save = img_copy

            image_to_save.save(buffer, format=fmt, quality=quality, optimize=True)
            image_bytes = buffer.getvalue()

            if len(image_bytes) <= max_size_bytes:
                mime_type = f"image/{fmt.lower()}"
                encoded_string = base64.b64encode(image_bytes).decode("utf-8")
                return f"data:{mime_type};base64,{encoded_string}"

    # If everything fails, raise error
    raise ValueError(
        f"Unable to compress image to under {max_size_bytes // (1024 * 1024)}MB. Please try a smaller image."
    )


def process_qwen_image_edit(image, confidence=0.5):
    """Process image using Qwen-Image-Edit model for object detection."""
    if not dashscope_available:
        st.error(get_translation("dashscope_not_available"))
        return None

    try:
        # Get API key from session state
        api_key = st.session_state.get("dashscope_api_key")
        if not api_key:
            st.error(get_translation("no_dashscope_key"))
            return None

        # Show compression status
        with st.spinner(get_translation("compressing_image")):
            # Encode image to base64 with compression
            image_base64 = encode_image_to_base64(image)

            # Calculate and show file size info
            base64_size = len(image_base64)
            original_size = len(image_base64) * 0.75  # Approximate original size
            st.info(
                get_translation("image_compressed", original_size=original_size / (1024 * 1024), base64_size=base64_size / (1024 * 1024))
            )

        # Create prompt for YOLO-like object detection
        prompt = f"""Create a YOLO-style annotated version of the given image.
Instructions:
1.Detect all objects in the image.
2.Draw bounding boxes around each detected object.
3.Display class labels at the top of each bounding box.
4.Add confidence scores next to each label, formatted as a decimal between 0.00 and 1.00.
5.Apply a confidence threshold of {confidence} — only show detections above this threshold.
6.Use distinct, easily distinguishable colors for different object classes.
7.Ensure that bounding boxes, labels, and confidence scores are clearly visible and readable, regardless of background.
8.Output: A new image with all annotations clearly drawn and labeled.
Goal:
Generate a clean, high-contrast annotated image showing YOLO-style detections with readable text and visually distinct bounding boxes for each class."""

        # Prepare messages for API
        messages = [
            {"role": "user", "content": [{"image": image_base64}, {"text": prompt}]}
        ]

        # Call Qwen-Image-Edit API with progress indication
        with st.spinner(get_translation("processing_with_qwen")):
            response = MultiModalConversation.call(
                api_key=api_key,
                model="qwen-image-edit",
                messages=messages,
                result_format="message",
                stream=False,
                watermark=False,
                negative_prompt="",
            )

        if response.status_code == 200:
            # Extract image URL from response
            content = response.output.choices[0].message.content
            if isinstance(content, list) and len(content) > 0 and "image" in content[0]:
                image_url = content[0]["image"]

                # Download the processed image
                with st.spinner(get_translation("downloading_image")):
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        # Save to temporary file and return as PIL Image
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".png"
                        ) as f:
                            f.write(img_response.content)
                            temp_path = f.name

                        processed_image = Image.open(temp_path)
                        os.unlink(temp_path)  # Clean up temp file
                        return processed_image
                    else:
                        st.error(
                            get_translation("download_failed", status=img_response.status_code)
                        )
                        return None
            else:
                st.error(get_translation("no_image_response"))
                return None
        else:
            st.error(
                get_translation("api_request_failed", status=response.status_code, message=response.message)
            )
            return None

    except ValueError as e:
        if "Unable to compress image" in str(e):
            st.error(get_translation("image_too_large"))
        else:
            st.error(get_translation("image_processing_error", error=str(e)))
        return None
    except Exception as e:
        st.error(get_translation("qwen_processing_error", error=str(e)))
        return None


def process_gemini_image(image, confidence=0.5):
    """Process image using Gemini 2.5 Flash Image model for object detection with image generation."""
    temp_image_path = None
    try:
        # Get API key from session state
        api_key = st.session_state.get("openrouter_api_key")
        if not api_key:
            st.error(get_translation("no_openrouter_key"))
            return None

        # Save image to temporary file for proper encoding
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            image.save(temp_file.name, format="PNG")
            temp_image_path = temp_file.name

        # Create prompt for annotated image generation
        prompt = f"""Create a YOLO-style annotated version of the given image.
Instructions:
1.Detect all objects in the image.
2.Draw bounding boxes around each detected object.
3.Display class labels at the top of each bounding box.
4.Add confidence scores next to each label, formatted as a decimal between 0.00 and 1.00.
5.Apply a confidence threshold of {confidence} — only show detections above this threshold.
6.Use distinct, easily distinguishable colors for different object classes.
7.Ensure that bounding boxes, labels, and confidence scores are clearly visible and readable, regardless of background.
8.Output: A new image with all annotations clearly drawn and labeled.
Goal:
Generate a clean, high-contrast annotated image showing YOLO-style detections with readable text and visually distinct bounding boxes for each class."""

        with st.spinner(get_translation("processing_with_gemini")):
            # Use direct requests for image generation modalities
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://testing.netlify.app/",  # Your app's URL
                "X-Title": "YOLO11 Object Detection",
            }

            # Read image and encode to base64
            with open(temp_image_path, "rb") as img_file:
                image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode()

            payload = {
                "model": "google/gemini-2.5-flash-image",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "modalities": ["image", "text"],
                "max_tokens": 1024
            }

            response = requests.post(url, headers=headers, json=payload)
            result = response.json()

            if response.status_code == 200 and result.get("choices"):
                message = result["choices"][0]["message"]

                # Check for generated annotated images
                if message.get("images"):
                    for image_data in message["images"]:
                        if image_data.get("image_url", {}).get("url"):
                            generated_image_url = image_data["image_url"]["url"]

                            # Handle both HTTP URLs and data URLs
                            if generated_image_url.startswith("data:"):
                                # Extract base64 data from data URL
                                base64_match = re.search(r'base64,(.+)', generated_image_url)
                                if base64_match:
                                    base64_data = base64_match.group(1)
                                    image_bytes = base64.b64decode(base64_data)
                                    annotated_image = Image.open(BytesIO(image_bytes))

                                    # Store any text response for additional info
                                    if message.get("content"):
                                        st.session_state.last_gemini_response = message["content"]

                                    st.success("✅ Gemini 2.5 Flash Image generated annotated image successfully!")
                                    return annotated_image
                            else:
                                # Download from HTTP URL
                                img_response = requests.get(generated_image_url)
                                if img_response.status_code == 200:
                                    # Convert to PIL Image
                                    annotated_image = Image.open(BytesIO(img_response.content))

                                    # Store any text response for additional info
                                    if message.get("content"):
                                        st.session_state.last_gemini_response = message["content"]

                                    st.success("✅ Gemini 2.5 Flash Image generated annotated image successfully!")
                                    return annotated_image

                # If no image generated, check for text analysis as fallback
                if message.get("content"):
                    st.session_state.last_gemini_response = message["content"]
                    st.info("Gemini provided text analysis instead of annotated image.")
                    return image
                else:
                    st.error(get_translation("no_response"))
                    return None
            else:
                # Display detailed error information
                error_detail = result.get('error', {})
                error_msg = error_detail.get('message', 'Unknown error')
                error_code = error_detail.get('code', 'Unknown code')

                st.error(get_translation("modalities_error", error_code=error_code, error_msg=error_msg))

                # Modalities error - return None since fallback is removed
                if "modalities" in str(error_msg).lower():
                    st.error(get_translation("modalities_not_available"))
                    return None
                else:
                    return None

    except Exception as e:
        st.error(get_translation("gemini_processing_error", error=str(e)))
        return None
    finally:
        # Clean up temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)




def capture_camera_frame():
    """Capture a single frame from the front camera."""
    # Try different camera indices to find the front camera
    camera_indices = [0, 1, 2]  # Try multiple camera indices

    for camera_idx in camera_indices:
        try:
            cap = cv2.VideoCapture(camera_idx)

            # Check if camera opened successfully
            if not cap.isOpened():
                continue

            # Set camera properties for better compatibility
            cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
            )  # For macOS compatibility
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Give camera time to initialize
            time.sleep(0.1)

            # Try to read a frame
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None and frame.size > 0:
                # Convert BGR to RGB for PIL and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            else:
                continue

        except Exception:
            continue

    return None


# Set page configuration
st.set_page_config(
    page_title=get_translation("page_title"),
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Language switcher in top right corner
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("EN" if st.session_state.language == "zh" else "中文", key="language_switcher"):
        st.session_state.language = "zh" if st.session_state.language == "en" else "en"
        st.rerun()

# Title and description
st.title(get_translation("title"))
st.markdown(get_translation("description"))

# Sidebar for inputs
with st.sidebar:
    st.header(get_translation("input"))
    input_method_options = [get_translation("upload_image_video"), get_translation("url_input"), get_translation("front_camera")]
    input_method = st.radio(
        get_translation("input_method"),
        input_method_options,
        index=st.session_state.get("input_method_index", 0),
        key="input_method_radio"
    )

    # Update session state when input method changes
    current_index = input_method_options.index(input_method)
    if current_index != st.session_state.get("input_method_index", 0):
        st.session_state["input_method_index"] = current_index

    if input_method == get_translation("upload_image_video"):
        # Deactivate camera when switching to other input methods
        st.session_state.camera_active = False
        st.session_state.camera_frame = None

        uploaded_file = st.file_uploader(
            get_translation("upload_file"),
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"],
        )
        if uploaded_file:
            uploaded_bytes = uploaded_file.getvalue()
            # Determine if it's an image or video based on file extension
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                if st.session_state.get("uploaded_image_bytes") != uploaded_bytes:
                    new_image = Image.open(BytesIO(uploaded_bytes)).copy()
                    st.session_state["current_image"] = new_image
                    st.session_state["current_video"] = None
                    st.session_state["uploaded_image_bytes"] = uploaded_bytes
                    st.session_state["uploaded_video_bytes"] = None
                    st.session_state.pop("results", None)
                    st.session_state.pop("annotated_img", None)
                    st.session_state.pop("output_video_path", None)
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                if st.session_state.get("uploaded_video_bytes") != uploaded_bytes:
                    # Save video to temp file for processing with correct extension
                    original_ext = os.path.splitext(uploaded_file.name)[1]
                    temp_video = tempfile.NamedTemporaryFile(
                        delete=False, suffix=original_ext
                    )
                    temp_video.write(uploaded_bytes)
                    temp_video.close()
                    st.session_state["current_video"] = temp_video.name
                    st.session_state["current_image"] = None
                    st.session_state["uploaded_video_bytes"] = uploaded_bytes
                    st.session_state["uploaded_image_bytes"] = None
                    st.session_state.pop("results", None)
                    st.session_state.pop("annotated_img", None)
                    st.session_state.pop("output_video_path", None)
            else:
                st.error(get_translation("unsupported_file"))
        else:
            st.session_state["uploaded_image_bytes"] = None
            st.session_state["uploaded_video_bytes"] = None

    elif input_method == get_translation("url_input"):
        # Deactivate camera when switching to other input methods
        st.session_state.camera_active = False
        st.session_state.camera_frame = None

        url = st.text_input(get_translation("enter_url"))
        if url and st.button(get_translation("load_image")):
            try:
                response = requests.get(url)
                response.raise_for_status()
                new_image = Image.open(BytesIO(response.content)).copy()
                st.session_state["current_image"] = new_image
                st.session_state["current_video"] = None
                st.session_state["uploaded_image_bytes"] = None
                st.session_state["uploaded_video_bytes"] = None
                st.session_state.pop("results", None)
                st.session_state.pop("annotated_img", None)
                st.session_state.pop("output_video_path", None)
                st.success(get_translation("upload_success"))
            except Exception as e:
                st.error(f"{get_translation('error_loading')}: {e}")

    elif input_method == get_translation("front_camera"):
        # Auto-activate camera when Front Camera is selected
        st.session_state.camera_active = True
        st.session_state.current_image = None
        st.session_state.current_video = None
        st.markdown(f"**{get_translation('camera_mode')}**")
        st.info(get_translation("camera_info"))

    # Model Settings
    st.header(get_translation("model_settings"))

    # Prepare model options with Qwen-Image-Edit and Gemini 2.5 Flash Image
    yolo_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
    additional_models = []
    if dashscope_available:
        additional_models.append("qwen-image-edit")
    additional_models.append("gemini-2.5-flash-image")

    if additional_models:
        model_options = yolo_models + additional_models
        help_text = get_translation("model_help")
    else:
        model_options = yolo_models
        help_text = get_translation("model_help_simple")

    model_name = st.selectbox(
        get_translation("select_model"),
        model_options,
        index=4,  # Always default to yolo11x.pt (index 4 in yolo_models list)
        help=help_text,
    )

    # Warn if cloud models are selected with video input
    if model_name == "qwen-image-edit" and st.session_state.get("current_video"):
        st.warning(get_translation("qwen_video_warning"))
    elif model_name == "gemini-2.5-flash-image" and st.session_state.get("current_video"):
        st.warning(get_translation("gemini_video_warning"))
    confidence = st.slider(get_translation("confidence_threshold"), 0.1, 0.9, 0.5, 0.05)

    # API Key configuration for cloud models
    if model_name in ["qwen-image-edit", "gemini-2.5-flash-image"]:
        st.subheader(get_translation("api_configuration"))

        if model_name == "qwen-image-edit":
            st.info(get_translation("dashscope_info"))
            api_key = st.text_input(get_translation("dashscope_key"), type="password")
            if api_key:
                st.session_state.dashscope_api_key = api_key
                st.success(get_translation("api_key_saved"))

        elif model_name == "gemini-2.5-flash-image":
            st.info(get_translation("gemini_api_info"))
            openrouter_key = st.text_input(
                get_translation("openrouter_key"),
                value=st.session_state.get("openrouter_api_key", ""),
                type="password"
            )
            if openrouter_key != st.session_state.get("openrouter_api_key", ""):
                st.session_state.openrouter_api_key = openrouter_key
                if openrouter_key:
                    st.success(get_translation("openrouter_key_saved"))

    # Device detection (for YOLO models only)
    if model_name == "qwen-image-edit":
        device_display = get_translation("qwen_device")
        st.info(get_translation("qwen_device_info"))
    elif model_name == "gemini-2.5-flash-image":
        device_display = get_translation("gemini_device")
        st.info(get_translation("gemini_device_info"))
    else:
        if torch.backends.mps.is_available():
            device = "mps"
            device_display = "MPS (Apple Silicon)"
        else:
            device = "cpu"
            device_display = "CPU"
        st.info(get_translation("detected_device", device_display=device_display))

    # Start detection button
    if st.button(get_translation("start_detection")):
        current_image = st.session_state.get("current_image")
        current_video = st.session_state.get("current_video")
        camera_active = st.session_state.get("camera_active", False)

        if current_image is None and current_video is None and not camera_active:
            st.error(get_translation("upload_error"))
        elif model_name in ["qwen-image-edit", "gemini-2.5-flash-image"] and current_video is not None:
            model_display = model_name.replace("-", " ").title()
            st.error(get_translation("qwen_image_only", model_display=model_display))
        elif camera_active:
            # Check if we have a camera frame captured via camera input
            if st.session_state.get("camera_frame"):
                if model_name == "qwen-image-edit":
                    with st.spinner(get_translation("qwen_camera_processing")):
                        processed_img = process_qwen_image_edit(
                            st.session_state.camera_frame, confidence
                        )
                        if processed_img:
                            st.session_state.annotated_img = np.array(processed_img)
                            # For Qwen, we don't have structured results, so we'll set a flag
                            st.session_state.qwen_processed = True
                            st.session_state.results = None
                elif model_name == "gemini-2.5-flash-image":
                        processed_img = process_gemini_image(
                            st.session_state.camera_frame, confidence
                        )
                        if processed_img:
                            st.session_state.annotated_img = np.array(processed_img)
                            # For Gemini, we don't have structured results, so we'll set a flag
                            st.session_state.gemini_processed = True
                            st.session_state.results = None
                        else:
                            # If no annotated image was generated, use the original camera frame
                            st.session_state.annotated_img = np.array(st.session_state.camera_frame)
                            st.session_state.gemini_processed = True
                            st.session_state.results = None
                else:
                    with st.spinner(get_translation("loading_model")):
                        model = YOLO(model_name)
                        img_array = np.array(st.session_state.camera_frame)
                        results = model(img_array, conf=confidence, device=device)
                        st.session_state.results = results
                        st.session_state.annotated_img = results[0].plot()
                        st.session_state.qwen_processed = False
                        st.session_state.gemini_processed = False
            else:
                st.warning(get_translation("camera_warning"))
        elif current_video is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(get_translation("processing_video"))

            # Run YOLO predict using subprocess to capture progress
            predict_code = f"""
from ultralytics import YOLO
model = YOLO('{model_name}')
model.predict(
    '{current_video}',
    save=True,
    project='.',
    name='detect',
    conf={confidence},
    device='{device}',
    exist_ok=True,
    verbose=True
)
"""
            cmd = [sys.executable, "-c", predict_code]

            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            total_frames = None
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                output = line.strip()
                current_frame, total, prog = parse_yolo_progress(output)
                if current_frame is not None:
                    if total_frames is None:
                        total_frames = total
                        progress_bar.progress(0.1)  # After initializing
                    scaled_progress = 0.1 + (prog / 100) * 0.8
                    progress_bar.progress(scaled_progress)
                    status_text.text(
                        get_translation("processing_frame", current_frame=current_frame, total_frames=total_frames, prog=prog)
                    )

            progress_bar.progress(0.95)
            status_text.text(get_translation("finalizing_video"))
            p.wait()

            # Find the output video path
            output_video_path = None
            try:
                # Look for any video file in the detect directory
                video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
                video_files = []

                if os.path.exists("detect"):
                    for file in os.listdir("detect"):
                        if any(file.lower().endswith(ext) for ext in video_extensions):
                            file_path = os.path.join("detect", file)
                            if os.path.isfile(file_path):
                                video_files.append(
                                    (file_path, os.path.getmtime(file_path))
                                )

                # Sort by modification time to get the most recent
                if video_files:
                    video_files.sort(key=lambda x: x[1], reverse=True)
                    output_video_path = video_files[0][0]
                    st.session_state.output_video_path = output_video_path
                    status_text.empty()
                    progress_bar.empty()
                else:
                    st.error(get_translation("video_error"))
            except Exception as e:
                st.error(get_translation("video_error_detail", error=e))
            finally:
                progress_bar.empty()
                status_text.empty()
        elif current_image is not None:
            if model_name == "qwen-image-edit":
                with st.spinner(get_translation("processing_with_qwen")):
                    processed_img = process_qwen_image_edit(current_image, confidence)
                    if processed_img:
                        st.session_state.annotated_img = np.array(processed_img)
                        # For Qwen, we don't have structured results, so we'll set a flag
                        st.session_state.qwen_processed = True
                        st.session_state.results = None
                        st.session_state.gemini_processed = False
            elif model_name == "gemini-2.5-flash-image":
                    processed_img = process_gemini_image(current_image, confidence)
                    if processed_img:
                        st.session_state.annotated_img = np.array(processed_img)
                        # For Gemini, we don't have structured results, so we'll set a flag
                        st.session_state.gemini_processed = True
                        st.session_state.results = None
                        st.session_state.qwen_processed = False
                    else:
                        # If no annotated image was generated, use the original image
                        st.session_state.annotated_img = np.array(current_image)
                        st.session_state.gemini_processed = True
                        st.session_state.results = None
                        st.session_state.qwen_processed = False
            else:
                with st.spinner(get_translation("loading_model")):
                    model = YOLO(model_name)
                    img_array = np.array(current_image)
                    results = model(img_array, conf=confidence, device=device)

                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.annotated_img = results[0].plot()
                    st.session_state.qwen_processed = False
                    st.session_state.gemini_processed = False


# Main area for results
current_image = st.session_state.get("current_image")
current_video = st.session_state.get("current_video")

if current_image is not None:
    # Check if we have detection results to hide input image
    has_results = ("results" in st.session_state or
                  st.session_state.get("qwen_processed", False) or
                  st.session_state.get("gemini_processed", False))

    if not has_results:
        st.subheader(get_translation("input_image"))
        st.image(current_image, use_container_width=True)

    # Create three columns for rotation buttons (only show when no results)
    if not has_results:
        col1, col2, col3 = st.columns([2, 9, 2])  # middle column is flexible spacer

        with col1:
            if st.button(get_translation("rotate_left")):
                rotated_image = current_image.rotate(90, expand=True)
                st.session_state["current_image"] = rotated_image
                if "results" in st.session_state:
                    del st.session_state.results
                if "annotated_img" in st.session_state:
                    del st.session_state.annotated_img
                st.rerun()

        with col3:
            if st.button(get_translation("rotate_right")):
                rotated_image = current_image.rotate(-90, expand=True)
                st.session_state["current_image"] = rotated_image
                if "results" in st.session_state:
                    del st.session_state.results
                if "annotated_img" in st.session_state:
                    del st.session_state.annotated_img
                st.rerun()

    if "results" in st.session_state or st.session_state.get("qwen_processed", False) or st.session_state.get("gemini_processed", False):
        annotated_img = st.session_state.annotated_img

        st.subheader(get_translation("detection_results"))
        st.image(
            annotated_img,
            caption=get_translation("annotated_caption"),
            use_container_width=True,
        )

        # Check if this is a YOLO result, Qwen result, or Gemini result
        if st.session_state.get("qwen_processed", False):
            st.success(get_translation("qwen_completed"))
            st.info(get_translation("qwen_info"))

            # Download button for Qwen result
            annotated_pil = Image.fromarray(annotated_img)
            buf = BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                get_translation("download_qwen"),
                buf.getvalue(),
                "qwen_detected_image.png",
                "image/png",
            )
        elif st.session_state.get("gemini_processed", False):
            st.success(get_translation("gemini_completed"))

            # Check if we have an annotated image or just text analysis
            has_annotated_image = st.session_state.get("last_gemini_response") is None or "analysis" not in st.session_state.get("last_gemini_response", "").lower()

            if has_annotated_image and not st.session_state.get("last_gemini_response"):
                st.info(get_translation("gemini_annotated_info"))
                # Download button for annotated result
                annotated_pil = Image.fromarray(annotated_img)
                buf = BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    get_translation("download_gemini"),
                    buf.getvalue(),
                    "gemini_annotated_image.png",
                    "image/png",
                )
            else:
                st.info(get_translation("gemini_analysis_info"))

                # Display the text analysis if available
                if st.session_state.get("last_gemini_response"):
                    st.subheader(get_translation("detection_analysis"))
                    st.text_area(get_translation("gemini_analysis"),
                               value=st.session_state.last_gemini_response,
                               height=200,
                               disabled=True)

                # Download button for original image
                annotated_pil = Image.fromarray(annotated_img)
                buf = BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    get_translation("download_original"),
                    buf.getvalue(),
                    "gemini_original_image.png",
                    "image/png",
                )
        elif "results" in st.session_state and st.session_state.results:
            results = st.session_state.results
            if results[0].boxes:
                st.subheader(get_translation("summary_table"))
                boxes = results[0].boxes
                class_names = results[0].names
                data = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    data.append(
                        {
                            get_translation("object"): class_names[int(cls)],
                            get_translation("confidence"): round(float(conf), 2),
                            get_translation("bounding_box"): f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                        }
                    )
                st.table(data)

                # Download button
                annotated_pil = Image.fromarray(annotated_img)
                buf = BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    get_translation("download_annotated"),
                    buf.getvalue(),
                    "detected_image.png",
                    "image/png",
                )
            else:
                st.info(get_translation("no_objects"))

elif current_video is not None:
    st.subheader(get_translation("input_video"))
    st.video(current_video)

    if "output_video_path" in st.session_state:
        output_video_path = st.session_state.output_video_path
        if os.path.exists(output_video_path):
            st.subheader(get_translation("detection_results"))
            st.video(output_video_path)

            # Download button for video
            with open(output_video_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                get_translation("download_video"),
                video_bytes,
                "detected_video.mp4",
                "video/mp4",
            )
        else:
            st.info(get_translation("processing_long"))

elif st.session_state.get("camera_active", False):
    # Camera input section
    st.subheader("📷 Camera Input")

    # Use Streamlit's built-in camera input
    camera_image = st.camera_input(get_translation("take_photo"))

    if camera_image:
        # Convert the uploaded image to PIL Image
        pil_image = Image.open(camera_image)
        st.session_state.camera_frame = pil_image
        st.success(get_translation("photo_captured"))
        st.image(pil_image, use_container_width=True, caption=get_translation("captured_photo"))
    else:
        st.info(get_translation("camera_capture_info"))

    # Display camera detection results if available
    if (
        "results" in st.session_state
        or st.session_state.get("qwen_processed", False)
        or st.session_state.get("gemini_processed", False)
    ) and st.session_state.get("camera_frame"):
        st.subheader(get_translation("camera_detection_results"))
        annotated_img = st.session_state.annotated_img
        st.image(
            annotated_img,
            caption=get_translation("object_detection_results"),
            use_container_width=True,
        )

        # Check if this is a Qwen result, Gemini result, or YOLO result
        if st.session_state.get("qwen_processed", False):
            st.success(get_translation("qwen_camera_completed"))
            st.info(get_translation("qwen_camera_info"))

            # Download button for Qwen camera result
            annotated_pil = Image.fromarray(annotated_img)
            buf = BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                get_translation("download_qwen_camera"),
                buf.getvalue(),
                "qwen_camera_detected.png",
                "image/png",
            )
        elif st.session_state.get("gemini_processed", False):
            st.success(get_translation("gemini_camera_completed"))

            # Check if we have an annotated image or just text analysis
            has_annotated_image = st.session_state.get("last_gemini_response") is None or "analysis" not in st.session_state.get("last_gemini_response", "").lower()

            if has_annotated_image and not st.session_state.get("last_gemini_response"):
                st.info(get_translation("gemini_camera_info"))
                # Download button for annotated camera result
                annotated_pil = Image.fromarray(annotated_img)
                buf = BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    get_translation("download_gemini_camera"),
                    buf.getvalue(),
                    "gemini_camera_annotated.png",
                    "image/png",
                )
            else:
                st.info(get_translation("gemini_camera_analysis_info"))

                # Display the text analysis if available
                if st.session_state.get("last_gemini_response"):
                    st.subheader(get_translation("camera_detection_analysis"))
                    st.text_area(get_translation("gemini_camera_analysis"),
                               value=st.session_state.last_gemini_response,
                               height=150,
                               disabled=True)

                # Download button for original camera image
                annotated_pil = Image.fromarray(annotated_img)
                buf = BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    get_translation("download_camera_image"),
                    buf.getvalue(),
                    "gemini_camera_original.png",
                    "image/png",
                )
        elif "results" in st.session_state and st.session_state.results:
            # Display detection summary for YOLO results
            if st.session_state.results[0].boxes:
                results = st.session_state.results
                boxes = results[0].boxes
                class_names = results[0].names
                data = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = box.cls[0]
                    data.append(
                        {
                            get_translation("object"): class_names[int(cls)],
                            get_translation("confidence"): round(float(conf), 2),
                            get_translation("bounding_box"): f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                        }
                    )
                st.table(data)

                # Download button for YOLO camera detection
                annotated_pil = Image.fromarray(annotated_img)
                buf = BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    get_translation("download_camera_detection"),
                    buf.getvalue(),
                    "camera_detected.png",
                    "image/png",
                )
            else:
                st.info(get_translation("no_detections"))

else:
    st.info(get_translation("no_media"))
