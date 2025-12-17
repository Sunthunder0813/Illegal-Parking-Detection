#!/usr/bin/env python3
# Raspberry Pi 5 + Hailo-8L
# Vehicle detection with multi-camera support

# =============================================================================
# REQUIRED SUDO COMMANDS FOR RASPBERRY PI 5 SETUP
# =============================================================================
# 1. Update package list:
#    sudo apt update
#
# 2. Install Hailo software and dependencies:
#    sudo apt install hailo-all
#
# 3. (Optional but recommended) Reboot after Hailo install:
#    sudo reboot
#
# 4. (If you need ffmpeg for RTSP testing)
#    sudo apt install ffmpeg
#
# 5. (If you want to use pip for Python dependencies)
#    sudo apt install python3-pip
#
# 6. (If you want to use git for GitHub push)
#    sudo apt install git
# =============================================================================

# =============================================================================
# FIX FOR SEGMENTATION FAULT AND QT ERRORS
# =============================================================================
# Set environment variables BEFORE importing cv2
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force X11 instead of Wayland
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"  # Suppress Qt warnings

# =============================================================================
# WHAT YOU NEED TO DO (Raspberry Pi 5 + Hailo AI Kit Setup)
# =============================================================================
#
# STEP 1: INSTALL HAILO SOFTWARE ON RASPBERRY PI 5
# ------------------------------------------------
# Run these commands on your Raspberry Pi 5:
#
#   sudo apt update
#   sudo apt install hailo-all
#
# This installs:
#   - hailo-firmware (NPU firmware)
#   - hailo-pcie-driver (PCIe driver for Hailo-8L)
#   - hailo-tappas-core (runtime libraries)
#   - hailort (Hailo runtime)
#
# After installation, REBOOT your Pi:
#   sudo reboot
#
# STEP 2: VERIFY HAILO IS DETECTED
# --------------------------------
# Run this command to check if Hailo-8L is detected:
#
#   hailortcli fw-control identify
#
# You should see output like:
#   "Device: Hailo-8L, Firmware Version: X.X.X"
#
# If not detected, check:
#   - AI Kit is properly seated on the Pi
#   - PCIe is enabled in /boot/config.txt
#
# STEP 3: GET THE HEF FILE (CRITICAL!) - UPDATED INSTRUCTIONS
# ------------------------------------
# Run this command on your Raspberry Pi to download the HEF file:
#
#   wget -O /home/set-admin/Illegal-Parking-Detection/yolov8n.hef \
#     https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8l/yolov8n.hef
#
# OR check if Hailo already installed example models:
#   ls /usr/share/hailo-models/
#   # If yolov8n.hef exists there, copy it:
#   cp /usr/share/hailo-models/yolov8n.hef /home/set-admin/Illegal-Parking-Detection/
#
# STEP 4: PLACE THE HEF FILE
# --------------------------
# Copy the .hef file to:
#   /home/set-admin/Illegal-Parking-Detection/yolov8n.hef
#
# STEP 5: INSTALL PYTHON DEPENDENCIES
# -----------------------------------
# On your Raspberry Pi 5:
#
#   pip install opencv-python numpy
#   pip install hailo-platform  # Should be installed with hailo-all
#
# STEP 6: CONNECT YOUR CAMERAS
# ----------------------------
# Update the camera IPs below to match your RTSP cameras
# Test camera connection first with:
#   ffplay rtsp://admin:password@192.168.18.2:554/h264
#
# STEP 7: RUN THE SCRIPT
# ----------------------
#   python sample.py
#
# Press 'q' to quit
#
# STEP 8: SETUP GITHUB AUTHENTICATION (For auto-push feature)
# -----------------------------------------------------------
# Option A - SSH Key (Recommended):
#   ssh-keygen -t ed25519 -C "santanderjoseph13@gmail.com"
#   eval "$(ssh-agent -s)"
#   ssh-add ~/.ssh/id_ed25519
#   cat ~/.ssh/id_ed25519.pub  # Add this to GitHub Settings -> SSH Keys
#   git remote set-url origin git@github.com:Sunthunder0813/Illegal-Parking-Detection.git
#
# Option B - Personal Access Token:
#   1. GitHub.com -> Settings -> Developer settings -> Personal access tokens
#   2. Generate token with 'repo' scope
#   3. Run: git config --global credential.helper store
#   4. First push will ask for username and token (use token as password)
#
# =============================================================================

import cv2
import numpy as np
import threading
from queue import Queue
import sys
import time
import subprocess
import signal

# -----------------------------
# Graceful shutdown handler (prevents segfault on Ctrl+C)
# -----------------------------
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n⚠️ Shutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# -----------------------------
# GitHub Push Configuration
# -----------------------------
ENABLE_GITHUB_PUSH = True
PUSH_INTERVAL_SECONDS = 60
SAVE_DETECTION_IMAGES = True
MIN_DETECTIONS_TO_SAVE = 1

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from github_push import setup_git, push_detection_event, push_to_github, init_repo
    GITHUB_AVAILABLE = True
    if ENABLE_GITHUB_PUSH:
        setup_git()
        init_repo(PROJECT_DIR)
        print("✅ GitHub push enabled")
except ImportError:
    GITHUB_AVAILABLE = False
    print("⚠️ GitHub push module not found. Auto-push disabled.")

last_push_time = 0

# -----------------------------
# Check Hailo Platform
# -----------------------------
HAILO_AVAILABLE = False
try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
    HAILO_AVAILABLE = True
    print("✅ Hailo platform detected")
except ImportError:
    print("⚠️ Hailo platform library (hailo_platform) not found.")
    print("   RUN: sudo apt install hailo-all && sudo reboot")
    from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
PT_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.pt"
ONNX_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.onnx"
HEF_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.hef"

# -----------------------------
# Auto-download HEF if missing
# -----------------------------
def download_hef():
    """Attempt to download HEF from Hailo Model Zoo or copy from system"""
    print("🔍 Attempting to find/download HEF file...")
    
    # Option 1: Check system-installed models
    system_hef_paths = [
        "/usr/share/hailo-models/yolov8n.hef",
        "/usr/share/hailo/models/yolov8n.hef",
        "/opt/hailo/models/yolov8n.hef"
    ]
    
    for path in system_hef_paths:
        if os.path.isfile(path):
            print(f"✅ Found system HEF: {path}")
            try:
                import shutil
                shutil.copy(path, HEF_MODEL)
                print(f"✅ Copied to: {HEF_MODEL}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to copy: {e}")
    
    # Option 2: Try to download from Hailo Model Zoo
    hef_urls = [
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8l/yolov8n.hef",
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov8n_h8l.hef"
    ]
    
    for url in hef_urls:
        print(f"📥 Trying to download from: {url}")
        try:
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "-O", HEF_MODEL, url],
                timeout=120
            )
            if result.returncode == 0 and os.path.isfile(HEF_MODEL) and os.path.getsize(HEF_MODEL) > 1000:
                print(f"✅ Downloaded HEF successfully!")
                return True
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
    
    return False

# -----------------------------
# Export PT -> ONNX if missing
# -----------------------------
# NOTE: This step is only needed if you're compiling the HEF yourself
# The ONNX file is an intermediate format, NOT used by Hailo directly
if not os.path.isfile(ONNX_MODEL):
    print("⚠️ ONNX model not found, exporting from PT...")
    try:
        from ultralytics import YOLO
        model = YOLO(PT_MODEL)
        model.export(format="onnx", imgsz=640, dynamic=False)
        print(f"✅ Exported ONNX model: {ONNX_MODEL}")
    except Exception as e:
        print(f"❌ Failed to export ONNX: {e}")
        sys.exit(1)

# -----------------------------
# Check for HEF - Now with auto-download
# -----------------------------
if HAILO_AVAILABLE and not os.path.isfile(HEF_MODEL):
    print("❌ HEF file not found. Attempting auto-download...")
    if not download_hef():
        print("=" * 60)
        print("CRITICAL: Could not find or download HEF file!")
        print("")
        print("MANUAL DOWNLOAD - Run this command:")
        print("  wget -O " + HEF_MODEL + " \\")
        print("    https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8l/yolov8n.hef")
        print("")
        print("OR check Hailo's example models:")
        print("  ls /usr/share/hailo-models/")
        print("=" * 60)
        print("Falling back to CPU inference (WILL BE VERY SLOW!).")
        HAILO_AVAILABLE = False

# -----------------------------
# Model setup
# -----------------------------
target = None  # Initialize target to None

if HAILO_AVAILABLE and os.path.isfile(HEF_MODEL):
    try:
        hef = HEF(HEF_MODEL)
        params = VDevice.create_params()
        target = VDevice(params)
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        input_info = network_group.get_input_vstream_infos()[0]
        INPUT_HEIGHT = input_info.shape[1]
        INPUT_WIDTH = input_info.shape[2]
        print(f"✅ HEF loaded successfully. Input shape: ({INPUT_HEIGHT}, {INPUT_WIDTH})")
    except Exception as e:
        print(f"❌ Failed to load HEF or configure Hailo: {e}")
        print("Falling back to CPU.")
        HAILO_AVAILABLE = False
        from ultralytics import YOLO
        model = YOLO(PT_MODEL)
        target = None

if not HAILO_AVAILABLE:
    print("⚠️ Setting up CPU inference (Ultralytics YOLO).")
    print("   WARNING: CPU inference on Pi 5 is ~1-2 FPS. Get a HEF file for ~30+ FPS!")
    try:
        from ultralytics import YOLO
        model = YOLO(PT_MODEL)
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        print("   Make sure yolov8n.pt exists or run: pip install ultralytics")
        sys.exit(1)

# -----------------------------
# Detected classes (vehicles + person)
# -----------------------------
# COCO: 0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'
detected_classes = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# COCO class names (80 classes, index 0 is 'person')
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# -----------------------------
# Camera info (using placeholders from original script)
# -----------------------------
# TODO: UPDATE THESE VALUES FOR YOUR CAMERAS!
username = "admin"          # <-- Change to your camera's username
password = ""               # <-- Change to your camera's password
cameras = [
    {"ip": "192.168.18.2", "name": "Camera 1"},   # <-- Update IP addresses
    {"ip": "192.168.18.71", "name": "Camera 2"}   # <-- Update IP addresses
]
# NOTE: Test your camera URLs first with:
#   ffplay "rtsp://admin:password@192.168.18.2:554/h264"

frame_queues = [Queue(maxsize=1) for _ in cameras]
stop_threads = False

# -----------------------------
# Hailo preprocessing
# -----------------------------
# The global variables INPUT_HEIGHT, INPUT_WIDTH are now set in the Hailo setup block
# but initialized here for safety.
INPUT_HEIGHT = 640
INPUT_WIDTH = 640

def preprocess_frame(frame):
    # Resize the image to the network input size (e.g., 640x640)
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # DO NOT normalize or convert to float32!
    # Add batch dimension (HWC -> 1HWC)
    return np.expand_dims(rgb.astype(np.uint8), axis=0)

# -----------------------------
# YOLOv8 Decoding (proper, simplified)
# -----------------------------
def decode_yolov8(output, frame_shape, conf_thres=0.4):
    h, w = frame_shape[:2]
    detections = []

    preds = output.reshape(-1, 84)
    boxes = preds[:, :4]
    scores = preds[:, 4:5] * preds[:, 5:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    for i in range(len(confidences)):
        if confidences[i] > conf_thres and class_ids[i] in detected_classes:
            cx, cy, bw, bh = boxes[i]
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(confidences[i]),
                "class_id": int(class_ids[i])
            })
    return detections

# -----------------------------
# Simple NMS (Non-Maximum Suppression)
# -----------------------------
def nms(detections, iou_threshold=0.5):
    if not detections:
        return []
    boxes = np.array([d['bbox'] for d in detections])
    scores = np.array([d['conf'] for d in detections])
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        box_area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union_area = box_area_i + box_area_rest - inter_area
        iou = inter_area / (union_area + 1e-6)
        idxs = idxs[1:][iou < iou_threshold]
    return [detections[i] for i in keep]

# -----------------------------
# Hailo inference pipeline (create ONCE)
# -----------------------------
hailo_infer_pipeline = None
if HAILO_AVAILABLE and os.path.isfile(HEF_MODEL):
    try:
        hailo_infer_pipeline = InferVStreams(network_group, network_group_params)
        hailo_infer_pipeline.__enter__()
    except Exception as e:
        print(f"❌ Failed to create Hailo InferVStreams: {e}")
        hailo_infer_pipeline = None

def run_hailo_inference(frame):
    input_data = preprocess_frame(frame)
    input_dict = {network_group.get_input_vstream_infos()[0].name: input_data}
    output = hailo_infer_pipeline.infer(input_dict)
    output_name = list(output.keys())[0]
    raw_output = output[output_name][0]
    # Decode YOLOv8 output
    detections = decode_yolov8(raw_output, frame.shape, conf_thres=0.4)
    # Apply NMS
    detections = nms(detections, iou_threshold=0.5)
    return detections

def run_inference(frame):
    # Use Hailo for detection if available and HEF is loaded, else fallback to CPU YOLO
    try:
        if HAILO_AVAILABLE and target is not None and hailo_infer_pipeline is not None:
            return run_hailo_inference(frame)
        else:
            results = model(frame, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = COCO_CLASS_NAMES[cls_id] if cls_id < len(COCO_CLASS_NAMES) else "unknown"
                    if class_name == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({'bbox': (x1, y1, x2, y2), 'conf': float(box.conf[0]), 'class_id': cls_id, 'class_name': class_name})
                    elif cls_id in detected_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({'bbox': (x1, y1, x2, y2), 'conf': float(box.conf[0]), 'class_id': cls_id, 'class_name': class_name})
            return detections
    except Exception as e:
        print(f"⚠️ Inference failed: {e}")
        return []

# -----------------------------
# Camera threads
# -----------------------------
def camera_reader(cap, queue, cam_name):
    global stop_threads, shutdown_requested
    while not stop_threads and not shutdown_requested:
        try:
            ret, frame = cap.read()
            if ret:
                # Use a non-blocking queue put to keep frame latency low
                if queue.full():
                    try: queue.get_nowait()
                    except: pass
                queue.put(frame)
            else:
                time.sleep(0.05)
        except Exception as e:
            print(f"⚠️ Camera {cam_name} read error: {e}")
            time.sleep(0.1)

caps = []
threads = []
for i, cam in enumerate(cameras):
    # Ensure correct RTSP URL format
    rtsp_url = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
    
    # Use CAP_FFMPEG backend for better RTSP support
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Optimization: Reduce internal buffer size and set FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    if not cap.isOpened():
        print(f"❌ Cannot connect to {cam['name']} at {cam['ip']}")
    else:
        print(f"✅ Connected to {cam['name']}")
        t = threading.Thread(target=camera_reader, args=(cap, frame_queues[i], cam['name']), daemon=True)
        t.start()
        threads.append(t)
    caps.append(cap)

# -----------------------------
# Display window setup (with error handling)
# -----------------------------
display_available = False
try:
    cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
    display_available = True
    print("✅ Display window created")
except Exception as e:
    print(f"⚠️ Cannot create display window: {e}")
    print("   Running in headless mode (no display)")

last_frames = [None for _ in cameras]

print("🚀 Starting vehicle detection...")
print("   Press 'q' to quit (or Ctrl+C)")

# -----------------------------
# Main loop with proper error handling
# -----------------------------
try:
    while not shutdown_requested:
        frames = []
        current_time = time.time()
        
        for i, cam in enumerate(cameras):
            try:
                frame = frame_queues[i].get_nowait()
                last_frames[i] = frame.copy()
            except:
                frame = last_frames[i]

            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"No Signal - {cam['name']}", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                try:
                    detections = run_inference(frame)
                    
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        conf = det['conf']
                        label = detected_classes.get(det['class_id'], "Unknown")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.putText(frame, cam['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # GitHub Push Logic
                    if (ENABLE_GITHUB_PUSH and GITHUB_AVAILABLE and 
                        len(detections) >= MIN_DETECTIONS_TO_SAVE and
                        (current_time - last_push_time) >= PUSH_INTERVAL_SECONDS):
                        
                        frame_to_save = frame.copy() if SAVE_DETECTION_IMAGES else None
                        push_thread = threading.Thread(
                            target=push_detection_event,
                            args=(PROJECT_DIR, detections, cam['name'], frame_to_save, True),
                            daemon=True
                        )
                        push_thread.start()
                        last_push_time = current_time
                except Exception as e:
                    print(f"⚠️ Processing error: {e}")

            frames.append(frame)

        # Display only if available
        if display_available and frames and all(f is not None for f in frames):
            try:
                target_h = 480
                resized = [cv2.resize(f, (int(f.shape[1]*target_h/f.shape[0]), target_h)) for f in frames]
                combined = cv2.hconcat(resized)
                cv2.imshow("Vehicle Detection", combined)
            except Exception as e:
                print(f"⚠️ Display error: {e}")
                display_available = False

        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("👋 Quit requested...")
            break

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")
except Exception as e:
    print(f"❌ Main loop error: {e}")

# -----------------------------
# Cleanup (careful order to avoid segfault)
# -----------------------------
print("🧹 Cleaning up...")

stop_threads = True
shutdown_requested = True

# Wait for threads to finish
time.sleep(0.5)

# Release cameras first
for cap in caps:
    try:
        if cap.isOpened():
            cap.release()
    except:
        pass

# Destroy OpenCV windows
try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Required to actually close windows
except:
    pass

# Final push
if ENABLE_GITHUB_PUSH and GITHUB_AVAILABLE:
    print("📤 Final push to GitHub...")
    try:
        push_to_github(PROJECT_DIR, "Final detection results - session ended")
    except Exception as e:
        print(f"⚠️ Final push failed: {e}")

# Release Hailo last
if hailo_infer_pipeline is not None:
    try:
        hailo_infer_pipeline.__exit__(None, None, None)
    except Exception as e:
        print(f"⚠️ Failed to release Hailo InferVStreams: {e}")

if target is not None:
    try:
        target.release()
    except Exception as e:
        print(f"⚠️ Failed to release Hailo VDevice: {e}")

print("👋 Cleanup complete")

# Common reasons why detection may not work when using Hailo:
#
# 1. The HEF file (compiled model) output format may not match the postprocess_results() logic.
#    - Hailo models often output raw tensors that require custom decoding.
#    - The postprocess_results() function assumes a specific output layout (e.g., [x_center, y_center, width, height, conf, class_id]).
#    - If the actual output is different (e.g., shape, order, or number of outputs), detections will fail or be empty.
#
# 2. The HEF file may not be compatible with your input size or model version.
#    - Ensure the yolov8n.hef matches your expected input size (e.g., 640x640) and is for the correct YOLO version.
#
# 3. The Hailo model may require additional post-processing (e.g., NMS, anchor decoding) not present in your code.
#    - Some Hailo HEFs are compiled without post-processing, so you must implement YOLO decoding and NMS yourself.
#
# 4. The input preprocessing may not match the model's expectations.
#    - Check if the model expects NHWC or NCHW, RGB or BGR, and normalization range.
#
# 5. The detection confidence threshold may be too high.
#    - Try lowering the conf_threshold in postprocess_results().
#
# 6. The model may not be loaded or configured correctly (target, network_group, etc.).
#    - Check for errors in the Hailo initialization block.
#
# 7. The output tensor may be empty or have unexpected values.
#    - Print or inspect the raw_output in run_hailo_inference() to debug.

# To debug:
# - Print the shape and a sample of raw_output in run_hailo_inference().
# - Use `hailortcli parse-hef yolov8n.hef` to inspect the output tensor format.
# - Compare your postprocess_results() logic with the actual output format.
# - Try running the Hailo TAPPAS demo or example scripts to verify hardware and model.
