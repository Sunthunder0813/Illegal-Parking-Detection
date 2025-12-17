#!/usr/bin/env python3
# Raspberry Pi 5 + Hailo-8L
# Vehicle detection with multi-camera support

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
# Detection classes (COCO)
# -----------------------------
# YOLOv8 COCO classes we want to detect
# 0: person, 2: car, 7: truck
DETECTION_CLASSES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Lower confidence threshold for better detection
CONFIDENCE_THRESHOLD = 0.25  # Lowered from 0.35

# Debug mode - set to True to see detection info
DEBUG_DETECTIONS = True

# Simplified labels for display
def get_label(class_id):
    if class_id == 0:
        return "Person"
    elif class_id in [2, 7]:
        return "Vehicle"
    elif class_id in [3, 5]:
        return "Vehicle"
    return "Object"

# Colors for different classes (BGR)
COLORS = {
    "Person": (0, 255, 0),    # Green
    "Vehicle": (255, 0, 0),   # Blue
    "Object": (0, 255, 255)   # Yellow
}

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
    # Normalize pixel values (0-255 -> 0.0-1.0)
    normalized = rgb.astype(np.float32) / 255.0
    # Add batch dimension (HWC -> 1HWC)
    input_data = np.expand_dims(normalized, axis=0)
    return input_data

def postprocess_results(output, frame_shape, conf_threshold=None):
    """
    Post-process YOLOv8 output from Hailo.
    Handles multiple output formats.
    """
    if conf_threshold is None:
        conf_threshold = CONFIDENCE_THRESHOLD
        
    detections = []
    original_h, original_w = frame_shape[:2]
    
    # Handle different output shapes
    if output is None:
        return detections
    
    if DEBUG_DETECTIONS:
        print(f"📊 Output shape: {output.shape}, dtype: {output.dtype}")
        print(f"   Output min: {output.min():.4f}, max: {output.max():.4f}")
    
    # Flatten if needed
    if len(output.shape) > 2:
        output = output.reshape(-1, output.shape[-1])
    
    if len(output.shape) != 2:
        if DEBUG_DETECTIONS:
            print(f"⚠️ Unexpected output shape: {output.shape}")
        return detections
    
    # YOLOv8 output formats:
    # - Shape (84, 8400) - needs transpose
    # - Shape (8400, 84) - standard format
    
    if output.shape[0] == 84 and output.shape[1] > 84:
        # Transpose from (84, 8400) to (8400, 84)
        output = output.T
        if DEBUG_DETECTIONS:
            print(f"   Transposed to: {output.shape}")
    
    num_detections = output.shape[0]
    num_values = output.shape[1]
    
    if DEBUG_DETECTIONS:
        print(f"   Processing {num_detections} potential detections with {num_values} values each")
    
    detected_count = 0
    
    if num_values >= 84:
        # Standard YOLOv8 format: [x, y, w, h, class_scores...]
        for i, detection in enumerate(output):
            x_center, y_center, box_w, box_h = detection[:4]
            class_scores = detection[4:84]  # 80 COCO classes
            
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])
            
            # Check ALL detections above threshold for debugging
            if conf > conf_threshold:
                detected_count += 1
                
                if DEBUG_DETECTIONS and detected_count <= 5:
                    print(f"   Detection {detected_count}: class={cls_id}, conf={conf:.3f}, "
                          f"box=({x_center:.1f}, {y_center:.1f}, {box_w:.1f}, {box_h:.1f})")
                
                # Only keep our target classes
                if cls_id in DETECTION_CLASSES:
                    # Scale coordinates from model input size to original frame size
                    scale_x = original_w / INPUT_WIDTH
                    scale_y = original_h / INPUT_HEIGHT
                    
                    # Convert center format to corner format and scale
                    x1 = int((x_center - box_w / 2) * scale_x)
                    y1 = int((y_center - box_h / 2) * scale_y)
                    x2 = int((x_center + box_w / 2) * scale_x)
                    y2 = int((y_center + box_h / 2) * scale_y)
                    
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, original_w - 1))
                    y1 = max(0, min(y1, original_h - 1))
                    x2 = max(0, min(x2, original_w - 1))
                    y2 = max(0, min(y2, original_h - 1))
                    
                    if x2 > x1 + 5 and y2 > y1 + 5:  # Valid box with minimum size
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf,
                            'class_id': cls_id,
                            'label': get_label(cls_id)
                        })
    
    elif num_values >= 5:
        # Possibly [x, y, w, h, conf] or [x, y, w, h, conf, cls] format
        for detection in output:
            if num_values >= 6:
                x_center, y_center, box_w, box_h, conf, cls_id = detection[:6]
            else:
                x_center, y_center, box_w, box_h, conf = detection[:5]
                cls_id = 0  # Assume person if no class
            
            cls_id_int = int(cls_id)
            
            if conf > conf_threshold and cls_id_int in DETECTION_CLASSES:
                # Check if normalized (0-1) or pixel coordinates
                if x_center <= 1.0 and y_center <= 1.0 and box_w <= 1.0 and box_h <= 1.0:
                    x1 = int((x_center - box_w / 2) * original_w)
                    y1 = int((y_center - box_h / 2) * original_h)
                    x2 = int((x_center + box_w / 2) * original_w)
                    y2 = int((y_center + box_h / 2) * original_h)
                else:
                    scale_x = original_w / INPUT_WIDTH
                    scale_y = original_h / INPUT_HEIGHT
                    x1 = int((x_center - box_w / 2) * scale_x)
                    y1 = int((y_center - box_h / 2) * scale_y)
                    x2 = int((x_center + box_w / 2) * scale_x)
                    y2 = int((y_center + box_h / 2) * scale_y)
                
                x1 = max(0, min(x1, original_w - 1))
                y1 = max(0, min(y1, original_h - 1))
                x2 = max(0, min(x2, original_w - 1))
                y2 = max(0, min(y2, original_h - 1))
                
                if x2 > x1 + 5 and y2 > y1 + 5:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class_id': cls_id_int,
                        'label': get_label(cls_id_int)
                    })
    
    if DEBUG_DETECTIONS:
        print(f"   Found {detected_count} detections above threshold, {len(detections)} are target classes")
    
    return detections


def simple_nms(detections, iou_threshold=0.45):
    """Simple Non-Maximum Suppression"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        detections = [
            d for d in detections
            if iou(best['bbox'], d['bbox']) < iou_threshold
        ]
    
    return keep


def iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


# -----------------------------
# Inference functions
# -----------------------------
def run_inference(frame):
    """Run inference on frame using available backend"""
    global model
    
    try:
        if HAILO_AVAILABLE and network_group is not None:
            return run_hailo_inference(frame)
        elif model is not None:
            # CPU (Ultralytics YOLO) inference
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Detect persons and vehicles
                    if cls_id in DETECTION_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf,
                            'class_id': cls_id,
                            'label': get_label(cls_id)
                        })
                        
                        if DEBUG_DETECTIONS:
                            print(f"🎯 CPU Detection: {get_label(cls_id)} conf={conf:.2f} at ({x1},{y1})-({x2},{y2})")
            
            return detections
        else:
            print("⚠️ No inference backend available!")
            return []
    except Exception as e:
        print(f"⚠️ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_hailo_inference(frame):
    """Run inference using Hailo NPU"""
    try:
        original_h, original_w = frame.shape[:2]
        
        if DEBUG_DETECTIONS:
            print(f"\n🔄 Running Hailo inference on frame {original_w}x{original_h}")
        
        # Preprocess: resize to model input size
        resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Check if model expects uint8 or float32
        # Most Hailo models expect uint8 input
        input_data_uint8 = np.expand_dims(rgb, axis=0).astype(np.uint8)
        input_data_float = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
        
        # Try uint8 first (more common for Hailo)
        try:
            with InferVStreams(network_group, network_group_params) as infer_pipeline:
                input_dict = {input_info.name: input_data_uint8}
                output = infer_pipeline.infer(input_dict)
        except Exception as e:
            if DEBUG_DETECTIONS:
                print(f"   uint8 input failed, trying float32: {e}")
            with InferVStreams(network_group, network_group_params) as infer_pipeline:
                input_dict = {input_info.name: input_data_float}
                output = infer_pipeline.infer(input_dict)
        
        # Get output - handle different output formats
        detections = []
        for output_name, output_data in output.items():
            output_array = np.array(output_data).squeeze()
            
            if DEBUG_DETECTIONS:
                print(f"   Output '{output_name}': shape {output_array.shape}")
            
            # Process the output
            dets = postprocess_results(output_array, (original_h, original_w))
            detections.extend(dets)
        
        # Apply NMS across all outputs
        detections = simple_nms(detections, iou_threshold=0.45)
        
        if DEBUG_DETECTIONS and len(detections) > 0:
            print(f"✅ Final detections after NMS: {len(detections)}")
            for det in detections:
                print(f"   {det['label']}: conf={det['conf']:.2f}, bbox={det['bbox']}")
        
        return detections
    except Exception as e:
        print(f"⚠️ Hailo inference error: {e}")
        import traceback
        traceback.print_exc()
        return []


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
# Detection classes (COCO)
# -----------------------------
# YOLOv8 COCO classes we want to detect
# 0: person, 2: car, 7: truck
DETECTION_CLASSES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Lower confidence threshold for better detection
CONFIDENCE_THRESHOLD = 0.25  # Lowered from 0.35

# Debug mode - set to True to see detection info
DEBUG_DETECTIONS = True

# Simplified labels for display
def get_label(class_id):
    if class_id == 0:
        return "Person"
    elif class_id in [2, 7]:
        return "Vehicle"
    elif class_id in [3, 5]:
        return "Vehicle"
    return "Object"

# Colors for different classes (BGR)
COLORS = {
    "Person": (0, 255, 0),    # Green
    "Vehicle": (255, 0, 0),   # Blue
    "Object": (0, 255, 255)   # Yellow
}

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
    # Normalize pixel values (0-255 -> 0.0-1.0)
    normalized = rgb.astype(np.float32) / 255.0
    # Add batch dimension (HWC -> 1HWC)
    input_data = np.expand_dims(normalized, axis=0)
    return input_data

def postprocess_results(output, frame_shape, conf_threshold=None):
    """
    Post-process YOLOv8 output from Hailo.
    Handles multiple output formats.
    """
    if conf_threshold is None:
        conf_threshold = CONFIDENCE_THRESHOLD
        
    detections = []
    original_h, original_w = frame_shape[:2]
    
    # Handle different output shapes
    if output is None:
        return detections
    
    if DEBUG_DETECTIONS:
        print(f"📊 Output shape: {output.shape}, dtype: {output.dtype}")
        print(f"   Output min: {output.min():.4f}, max: {output.max():.4f}")
    
    # Flatten if needed
    if len(output.shape) > 2:
        output = output.reshape(-1, output.shape[-1])
    
    if len(output.shape) != 2:
        if DEBUG_DETECTIONS:
            print(f"⚠️ Unexpected output shape: {output.shape}")
        return detections
    
    # YOLOv8 output formats:
    # - Shape (84, 8400) - needs transpose
    # - Shape (8400, 84) - standard format
    
    if output.shape[0] == 84 and output.shape[1] > 84:
        # Transpose from (84, 8400) to (8400, 84)
        output = output.T
        if DEBUG_DETECTIONS:
            print(f"   Transposed to: {output.shape}")
    
    num_detections = output.shape[0]
    num_values = output.shape[1]
    
    if DEBUG_DETECTIONS:
        print(f"   Processing {num_detections} potential detections with {num_values} values each")
    
    detected_count = 0
    
    if num_values >= 84:
        # Standard YOLOv8 format: [x, y, w, h, class_scores...]
        for i, detection in enumerate(output):
            x_center, y_center, box_w, box_h = detection[:4]
            class_scores = detection[4:84]  # 80 COCO classes
            
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])
            
            # Check ALL detections above threshold for debugging
            if conf > conf_threshold:
                detected_count += 1
                
                if DEBUG_DETECTIONS and detected_count <= 5:
                    print(f"   Detection {detected_count}: class={cls_id}, conf={conf:.3f}, "
                          f"box=({x_center:.1f}, {y_center:.1f}, {box_w:.1f}, {box_h:.1f})")
                
                # Only keep our target classes
                if cls_id in DETECTION_CLASSES:
                    # Scale coordinates from model input size to original frame size
                    scale_x = original_w / INPUT_WIDTH
                    scale_y = original_h / INPUT_HEIGHT
                    
                    # Convert center format to corner format and scale
                    x1 = int((x_center - box_w / 2) * scale_x)
                    y1 = int((y_center - box_h / 2) * scale_y)
                    x2 = int((x_center + box_w / 2) * scale_x)
                    y2 = int((y_center + box_h / 2) * scale_y)
                    
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, original_w - 1))
                    y1 = max(0, min(y1, original_h - 1))
                    x2 = max(0, min(x2, original_w - 1))
                    y2 = max(0, min(y2, original_h - 1))
                    
                    if x2 > x1 + 5 and y2 > y1 + 5:  # Valid box with minimum size
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf,
                            'class_id': cls_id,
                            'label': get_label(cls_id)
                        })
    
    elif num_values >= 5:
        # Possibly [x, y, w, h, conf] or [x, y, w, h, conf, cls] format
        for detection in output:
            if num_values >= 6:
                x_center, y_center, box_w, box_h, conf, cls_id = detection[:6]
            else:
                x_center, y_center, box_w, box_h, conf = detection[:5]
                cls_id = 0  # Assume person if no class
            
            cls_id_int = int(cls_id)
            
            if conf > conf_threshold and cls_id_int in DETECTION_CLASSES:
                # Check if normalized (0-1) or pixel coordinates
                if x_center <= 1.0 and y_center <= 1.0 and box_w <= 1.0 and box_h <= 1.0:
                    x1 = int((x_center - box_w / 2) * original_w)
                    y1 = int((y_center - box_h / 2) * original_h)
                    x2 = int((x_center + box_w / 2) * original_w)
                    y2 = int((y_center + box_h / 2) * original_h)
                else:
                    scale_x = original_w / INPUT_WIDTH
                    scale_y = original_h / INPUT_HEIGHT
                    x1 = int((x_center - box_w / 2) * scale_x)
                    y1 = int((y_center - box_h / 2) * scale_y)
                    x2 = int((x_center + box_w / 2) * scale_x)
                    y2 = int((y_center + box_h / 2) * scale_y)
                
                x1 = max(0, min(x1, original_w - 1))
                y1 = max(0, min(y1, original_h - 1))
                x2 = max(0, min(x2, original_w - 1))
                y2 = max(0, min(y2, original_h - 1))
                
                if x2 > x1 + 5 and y2 > y1 + 5:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class_id': cls_id_int,
                        'label': get_label(cls_id_int)
                    })
    
    if DEBUG_DETECTIONS:
        print(f"   Found {detected_count} detections above threshold, {len(detections)} are target classes")
    
    return detections


def simple_nms(detections, iou_threshold=0.45):
    """Simple Non-Maximum Suppression"""
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        detections = [
            d for d in detections
            if iou(best['bbox'], d['bbox']) < iou_threshold
        ]
    
    return keep


def iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


# -----------------------------
# Inference functions
# -----------------------------
def run_inference(frame):
    """Run inference on frame using available backend"""
    global model
    
    try:
        if HAILO_AVAILABLE and network_group is not None:
            return run_hailo_inference(frame)
        elif model is not None:
            # CPU (Ultralytics YOLO) inference
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Detect persons and vehicles
                    if cls_id in DETECTION_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf,
                            'class_id': cls_id,
                            'label': get_label(cls_id)
                        })
                        
                        if DEBUG_DETECTIONS:
                            print(f"🎯 CPU Detection: {get_label(cls_id)} conf={conf:.2f} at ({x1},{y1})-({x2},{y2})")
            
            return detections
        else:
            print("⚠️ No inference backend available!")
            return []
    except Exception as e:
        print(f"⚠️ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_hailo_inference(frame):
    """Run inference using Hailo NPU"""
    try:
        original_h, original_w = frame.shape[:2]
        
        if DEBUG_DETECTIONS:
            print(f"\n🔄 Running Hailo inference on frame {original_w}x{original_h}")
        
        # Preprocess: resize to model input size
        resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Check if model expects uint8 or float32
        # Most Hailo models expect uint8 input
        input_data_uint8 = np.expand_dims(rgb, axis=0).astype(np.uint8)
        input_data_float = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)
        
        # Try uint8 first (more common for Hailo)
        try:
            with InferVStreams(network_group, network_group_params) as infer_pipeline:
                input_dict = {input_info.name: input_data_uint8}
                output = infer_pipeline.infer(input_dict)
        except Exception as e:
            if DEBUG_DETECTIONS:
                print(f"   uint8 input failed, trying float32: {e}")
            with InferVStreams(network_group, network_group_params) as infer_pipeline:
                input_dict = {input_info.name: input_data_float}
                output = infer_pipeline.infer(input_dict)
        
        # Get output - handle different output formats
        detections = []
        for output_name, output_data in output.items():
            output_array = np.array(output_data).squeeze()
            
            if DEBUG_DETECTIONS:
                print(f"   Output '{output_name}': shape {output_array.shape}")
            
            # Process the output
            dets = postprocess_results(output_array, (original_h, original_w))
            detections.extend(dets)
        
        # Apply NMS across all outputs
        detections = simple_nms(detections, iou_threshold=0.45)
        
        if DEBUG_DETECTIONS and len(detections) > 0:
            print(f"✅ Final detections after NMS: {len(detections)}")
            for det in detections:
                print(f"   {det['label']}: conf={det['conf']:.2f}, bbox={det['bbox']}")
        
        return detections
    except Exception as e:
        print(f"⚠️ Hailo inference error: {e}")
        import traceback
        traceback.print_exc()
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
                    
                    # Draw detections with new function
                    frame, counts = draw_detections(frame, detections, cam['name'])
                    
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
if target is not None:
    try:
        target.release()
    except Exception as e:
        print(f"⚠️ Failed to release Hailo VDevice: {e}")
        
print("👋 Cleanup complete")
