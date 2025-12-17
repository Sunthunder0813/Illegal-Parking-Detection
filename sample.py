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
# Use the full COCO HEF (person + vehicles + all classes)
HEF_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n_full.hef"  # <-- Make sure this matches the downloaded file

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
# Vehicle classes (using COCO subset)
# -----------------------------
# YOLOv8 default COCO classes: 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'
vehicle_classes = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}
# Add person class for display and labeling
person_class_id = 0
person_class_name = "Person"

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

def postprocess_results(output, frame_shape, conf_threshold=0.5):
    # DEBUG: Print output shape and a sample
    print(f"[DEBUG] Hailo output shape: {output.shape}")
    print(f"[DEBUG] Hailo output sample: {output.flatten()[:10]}")

    detections = []
    h, w = frame_shape[:2]

    # Typical YOLOv8 output: (N, 84) where 84 = 4 + 1 + 80
    # [x, y, w, h, obj_conf, class_conf_0, ..., class_conf_79]
    if len(output.shape) == 2 and output.shape[1] >= 85:
        for row in output:
            x, y, bw, bh = row[:4]
            obj_conf = row[4]
            class_confs = row[5:]
            cls_id = int(np.argmax(class_confs))
            cls_conf = class_confs[cls_id]
            conf = obj_conf * cls_conf

            # Only keep vehicle/person classes
            if conf > conf_threshold and (cls_id in vehicle_classes or cls_id == person_class_id):  # 0: person
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                detections.append({'bbox': (x1, y1, x2, y2), 'conf': float(conf), 'class_id': cls_id})
    else:
        # Fallback: try to parse as before (legacy)
        for detection in output:
            if len(detection) >= 6:
                x_center, y_center, width, height, conf, cls_id = detection[:6]
                cls_id_int = int(cls_id)
                if conf > conf_threshold and (cls_id_int in vehicle_classes or cls_id_int == person_class_id):
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'class_id': cls_id_int})
    return detections

def run_hailo_inference(frame):
    input_data = preprocess_frame(frame)
    
    # The input data for Hailo is expected to be BHW(C) for a typical NCHW input.
    # For a YOLO model, the input is usually BGR or RGB, NCHW (1, 3, 640, 640).
    # Since preprocess_frame creates (1, 640, 640, 3) (NHWC), we need to ensure the Hailo HEF is expecting NHWC
    # or transpose the NumPy array to NCHW before sending. Assuming HEF input expects NHWC or the
    # platform handles the conversion, but typically deep learning expects NCHW.
    # We will stick to the original HWC->1HWC logic in preprocess for now, as that's standard for Hailo.
    
    with InferVStreams(network_group, network_group_params) as infer_pipeline:
        input_dict = {network_group.get_input_vstream_infos()[0].name: input_data}
        output = infer_pipeline.infer(input_dict)
    
    # The output will be a dictionary of {output_vstream_name: np.array}
    output_name = list(output.keys())[0]
    raw_output = output[output_name][0]
    
    # The raw_output from the HEF needs to be interpreted. This is the most likely source of the SegFault/bad results
    # if the format is wrong. The format is typically a flat tensor (e.g., 84 x 8400) that needs special decoding.
    # Assuming the HEF was compiled with a custom post-processor or the raw output is simplified:
    
    return postprocess_results(raw_output, frame.shape)

def run_inference(frame):
    try:
        if HAILO_AVAILABLE:
            return run_hailo_inference(frame)
        else:
            # CPU (Ultralytics YOLO) inference
            results = model(frame, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    # Only detect vehicle and person classes
                    if cls_id in vehicle_classes or cls_id == person_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({'bbox': (x1, y1, x2, y2), 'conf': float(box.conf[0]), 'class_id': cls_id})
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
                        # Show "Person" for class 0, otherwise vehicle label
                        if det['class_id'] == person_class_id:
                            label = person_class_name
                        else:
                            label = vehicle_classes.get(det['class_id'], "Unknown")
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
if target is not None:
    try:
        target.release()
    except Exception as e:
        print(f"⚠️ Failed to release Hailo VDevice: {e}")
        
print("👋 Cleanup complete")
