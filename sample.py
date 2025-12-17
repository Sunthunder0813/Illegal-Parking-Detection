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
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                                 ConfigureParams, InputVStreamParams, OutputVStreamParams,
                                 FormatType)
    HAILO_AVAILABLE = True
    print("✅ Hailo platform detected")
except ImportError:
    print("❌ Hailo platform library not found.")
    print("   RUN: sudo apt install hailo-all && sudo reboot")
    print("   This script requires Hailo-8L for inference.")
    sys.exit(1)

# -----------------------------
# Paths
# -----------------------------
HEF_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.hef"

# -----------------------------
# Global variables
# -----------------------------
model = None
target = None
network_group = None
network_group_params = None
input_info = None
output_info = None
input_vstream_params = None
output_vstream_params = None
INPUT_HEIGHT = 640
INPUT_WIDTH = 640

# -----------------------------
# Detection settings
# -----------------------------
DETECTION_CLASSES = {
    0: "Person",  # YOLOv8n COCO: 0 is 'person'
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

CONFIDENCE_THRESHOLD = 0.15  # Lowered from 0.25 to 0.15 for more sensitivity
DEBUG_DETECTIONS = True

def get_label(class_id):
    if class_id == 0:
        return "Person"
    elif class_id in [2, 3, 5, 7]:
        return "Vehicle"
    return "Object"

COLORS = {
    "Person": (0, 255, 0),
    "Vehicle": (255, 0, 0),
    "Object": (0, 255, 255)
}

# -----------------------------
# Auto-download HEF if missing
# -----------------------------
def find_local_hef():
    """Find HEF file from local/system paths only (no URL download)"""
    print("🔍 Searching for local HEF file...")
    
    # Check if already at target location
    if os.path.isfile(HEF_MODEL):
        print(f"✅ Found HEF at: {HEF_MODEL}")
        return True
    
    # Check system paths where Hailo may have installed models
    system_hef_paths = [
        "/usr/share/hailo-models/yolov8n.hef",
        "/usr/share/hailo/models/yolov8n.hef",
        "/opt/hailo/models/yolov8n.hef",
        "/home/set-admin/yolov8n.hef",
        os.path.expanduser("~/yolov8n.hef"),
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
    
    return False

# -----------------------------
# Model Setup - HAILO ONLY
# -----------------------------
if not os.path.isfile(HEF_MODEL):
    print(f"⚠️ HEF model not found at: {HEF_MODEL}")
    if not find_local_hef():
        print("❌ Local HEF file not found.")
        print("   Please place yolov8n.hef at:")
        print(f"   {HEF_MODEL}")
        sys.exit(1)

try:
    hef = HEF(HEF_MODEL)
    params = VDevice.create_params()
    target = VDevice(params)
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    
    # Get input and output vstream info
    input_vstreams_info = network_group.get_input_vstream_infos()
    output_vstreams_info = network_group.get_output_vstream_infos()
    input_info = input_vstreams_info[0]
    output_info = output_vstreams_info
    
    # Create vstream params using the correct API
    input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    
    INPUT_HEIGHT = input_info.shape[1]
    INPUT_WIDTH = input_info.shape[2]
    print(f"✅ Hailo-8L initialized successfully!")
    print(f"   Model: {HEF_MODEL}")
    print(f"   Input shape: ({INPUT_HEIGHT}, {INPUT_WIDTH})")
    print(f"   Output layers: {[info.name for info in output_vstreams_info]}")
except Exception as e:
    print(f"❌ Hailo setup failed: {e}")
    print("   Make sure:")
    print("   1. Hailo AI Kit is properly connected")
    print("   2. Run: hailortcli fw-control identify")
    print("   3. Reboot after installing: sudo apt install hailo-all")
    sys.exit(1)

# -----------------------------
# Camera Configuration
# -----------------------------
username = "admin"
password = ""
cameras = [
    {"ip": "192.168.18.2", "name": "Camera 1"},
    {"ip": "192.168.18.71", "name": "Camera 2"}
]

frame_queues = [Queue(maxsize=1) for _ in cameras]
stop_threads = False

# -----------------------------
# Post-processing functions
# -----------------------------
def postprocess_results(output, frame_shape, conf_threshold=None):
    if conf_threshold is None:
        conf_threshold = CONFIDENCE_THRESHOLD
    detections = []
    original_h, original_w = frame_shape[:2]
    
    if output is None or len(output) == 0:
        return detections
    
    if len(output.shape) > 2:
        output = output.reshape(-1, output.shape[-1])
    
    if len(output.shape) != 2:
        return detections
    
    if output.shape[0] == 84 and output.shape[1] > 84:
        output = output.T
    
    num_values = output.shape[1]
    
    if num_values >= 84:
        for detection in output:
            x_center, y_center, box_w, box_h = detection[:4]
            class_scores = detection[4:84]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])
            
            # --- DEBUG: Print person confidence ---
            if DEBUG_DETECTIONS and cls_id == 0:
                print(f"[DEBUG] Person conf: {conf:.3f}")

            if conf > conf_threshold and cls_id in DETECTION_CLASSES:
                scale_x = original_w / INPUT_WIDTH
                scale_y = original_h / INPUT_HEIGHT
                x1 = int((x_center - box_w / 2) * scale_x)
                y1 = int((y_center - box_h / 2) * scale_y)
                x2 = int((x_center + box_w / 2) * scale_x)
                y2 = int((y_center + box_h / 2) * scale_y)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(original_w-1, x2), min(original_h-1, y2)
                
                if x2 > x1 + 5 and y2 > y1 + 5:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class_id': cls_id,
                        'label': get_label(cls_id)
                    })
    return detections

def simple_nms(detections, iou_threshold=0.45):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if iou(best['bbox'], d['bbox']) < iou_threshold]
    return keep

def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0

# -----------------------------
# Inference function - HAILO ONLY
# -----------------------------
def run_inference(frame):
    """Run inference using Hailo-8L NPU"""
    try:
        original_h, original_w = frame.shape[:2]
        resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # --- FIX: Convert to NCHW (channels first) ---
        nchw = np.transpose(rgb, (2, 0, 1))  # (3, H, W)
        input_data = np.expand_dims(nchw, axis=0).astype(np.uint8)  # (1, 3, H, W)
        
        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            output = infer_pipeline.infer({input_info.name: input_data})
        
        detections = []
        for output_data in output.values():
            output_array = np.array(output_data).squeeze()
            dets = postprocess_results(output_array, (original_h, original_w))
            detections.extend(dets)
        # --- DEBUG: Print all detections ---
        if DEBUG_DETECTIONS:
            print(f"[DEBUG] Total detections: {len(detections)}")
            for d in detections:
                print(f"[DEBUG] Det: {d}")
        return simple_nms(detections)
    except Exception as e:
        print(f"⚠️ Hailo inference error: {e}")
        return []

def draw_detections(frame, detections, cam_name):
    counts = {"Person": 0, "Vehicle": 0}
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det.get('label', 'Object')
        color = COLORS.get(label, (0, 255, 255))
        if label in counts:
            counts[label] += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {det['conf']:.2f}"
        cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return frame, counts

# -----------------------------
# Camera thread
# -----------------------------
def camera_reader(cap, queue, cam_name):
    global stop_threads, shutdown_requested
    while not stop_threads and not shutdown_requested:
        ret, frame = cap.read()
        if ret:
            if queue.full():
                try: queue.get_nowait()
                except: pass
            queue.put(frame)
        else:
            time.sleep(0.05)

# -----------------------------
# Initialize cameras
# -----------------------------
caps = []
threads = []
for i, cam in enumerate(cameras):
    rtsp_url = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        print(f"✅ Connected to {cam['name']}")
        t = threading.Thread(target=camera_reader, args=(cap, frame_queues[i], cam['name']), daemon=True)
        t.start()
        threads.append(t)
    else:
        print(f"❌ Cannot connect to {cam['name']}")
    caps.append(cap)

# -----------------------------
# Main loop
# -----------------------------
cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
last_frames = [None] * len(cameras)

print("🚀 Starting detection... Press 'q' to quit")

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
                cv2.putText(frame, f"No Signal - {cam['name']}", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                detections = run_inference(frame)
                frame, _ = draw_detections(frame, detections, cam['name'])
            
            frames.append(frame)
        
        if frames:
            target_h = 480
            resized = [cv2.resize(f, (int(f.shape[1]*target_h/f.shape[0]), target_h)) for f in frames]
            cv2.imshow("Vehicle Detection", cv2.hconcat(resized))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# Cleanup
stop_threads = True
time.sleep(0.3)
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
if target:
    target.release()
print("👋 Done")
