#!/usr/bin/env python3
# Raspberry Pi 5 + Hailo-8L
# Vehicle detection with multi-camera support

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
# STEP 3: GET THE HEF FILE (CRITICAL!)
# ------------------------------------
# The .hef file CANNOT be created on the Raspberry Pi!
# You have TWO options:
#
# OPTION A: Download pre-compiled HEF from Hailo Model Zoo (EASIEST)
#   1. Go to: https://github.com/hailo-ai/hailo_model_zoo/releases
#   2. Download yolov8n.hef for hailo8l
#   3. Or use hailo_model_zoo CLI:
#      pip install hailo_model_zoo
#      hailomz compile yolov8n --hw-arch hailo8l --har /path/to/save
#
# OPTION B: Compile yourself on x86 Ubuntu machine
#   1. Get an x86 Ubuntu 20.04/22.04 machine
#   2. Install Hailo Dataflow Compiler (DFC) from developer.hailo.ai
#   3. Transfer your yolov8n.onnx to that machine
#   4. Run:
#      hailo parser onnx yolov8n.onnx --hw-arch hailo8l
#      hailo optimize yolov8n.har
#      hailo compiler yolov8n.har --hw-arch hailo8l
#   5. Transfer resulting yolov8n.hef back to your Pi
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

import os
import cv2
import numpy as np
import threading
from queue import Queue
import sys
import time
import subprocess

# -----------------------------
# GitHub Push Configuration
# -----------------------------
# TODO: Set these options for automatic GitHub push
ENABLE_GITHUB_PUSH = True           # Set to True to enable auto-push
PUSH_INTERVAL_SECONDS = 60          # How often to push results (minimum seconds between pushes)
SAVE_DETECTION_IMAGES = True        # Save images with detections
MIN_DETECTIONS_TO_SAVE = 1          # Minimum detections to trigger a save

# Get the project directory (where this script is located)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import GitHub push module
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

# Track last push time
last_push_time = 0

# -----------------------------
# Check Hailo Platform
# -----------------------------
# TODO: If this import fails, you need to install hailo-all package (see STEP 1 above)
HAILO_AVAILABLE = False
try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
    HAILO_AVAILABLE = True
    print("✅ Hailo platform detected")
except ImportError:
    print("⚠️ Hailo platform library (hailo_platform) not found.")
    print("   RUN: sudo apt install hailo-all && sudo reboot")
    # Fallback uses Ultralytics YOLO library (CPU - VERY SLOW on Pi!)
    from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
# TODO: Make sure these paths exist on your Raspberry Pi
PT_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.pt"
ONNX_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.onnx"
HEF_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.hef"  # <-- YOU MUST PROVIDE THIS FILE!

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
# Check for HEF (Model must be pre-compiled on x86 machine)
# -----------------------------
# TODO: THIS IS THE MOST IMPORTANT STEP!
# You MUST have a .hef file to use Hailo acceleration
# See STEP 3 above for how to get it
if HAILO_AVAILABLE and not os.path.isfile(HEF_MODEL):
    print("❌ HEF file not found.")
    print("=" * 60)
    print("CRITICAL: You need to provide a .hef file!")
    print("")
    print("EASIEST OPTION - Download from Hailo Model Zoo:")
    print("  1. Visit: https://github.com/hailo-ai/hailo_model_zoo")
    print("  2. Download yolov8n HEF for hailo8l")
    print("  3. Copy to: " + HEF_MODEL)
    print("")
    print("OR use Hailo's example models:")
    print("  ls /usr/share/hailo-models/")
    print("=" * 60)
    print("Falling back to CPU inference (WILL BE VERY SLOW!).")
    HAILO_AVAILABLE = False


# -----------------------------
# Model setup
# -----------------------------
if HAILO_AVAILABLE and os.path.isfile(HEF_MODEL):
    try:
        # Load the pre-compiled HEF model
        hef = HEF(HEF_MODEL)
        
        # Create virtual device parameters
        params = VDevice.create_params()
        target = VDevice(params)
        
        # TODO: If you get errors here, check that:
        #   1. Hailo-8L is detected: hailortcli fw-control identify
        #   2. PCIe driver is loaded: lsmod | grep hailo
        #   3. Device exists: ls /dev/hailo*
        
        # Configure for PCIe interface (Raspberry Pi 5 AI Kit uses PCIe)
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        
        # Get model input dimensions from the HEF
        input_info = network_group.get_input_vstream_infos()[0]
        global INPUT_HEIGHT, INPUT_WIDTH
        INPUT_HEIGHT = input_info.shape[1]
        INPUT_WIDTH = input_info.shape[2]
        
        print(f"✅ HEF loaded successfully. Input shape: ({INPUT_HEIGHT}, {INPUT_WIDTH})")
    except Exception as e:
        print(f"❌ Failed to load HEF or configure Hailo: {e}")
        print("   Check: hailortcli fw-control identify")
        print("Falling back to CPU.")
        HAILO_AVAILABLE = False
        from ultralytics import YOLO
        model = YOLO(PT_MODEL)
        target = None
else:
    # CPU Fallback setup - WARNING: This will be VERY slow on Raspberry Pi!
    print("⚠️ Setting up CPU inference (Ultralytics YOLO).")
    print("   WARNING: CPU inference on Pi 5 is ~1-2 FPS. Get a HEF file for ~30+ FPS!")
    from ultralytics import YOLO
    model = YOLO(PT_MODEL)
    target = None

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
    # TODO: This postprocessing may need adjustment based on your specific HEF!
    # Different HEF compilations can have different output formats.
    # If detections aren't working, you may need to:
    #   1. Check the HEF output format with: hailortcli parse-hef yolov8n.hef
    #   2. Adjust this function to match the output tensor layout
    detections = []
    h, w = frame_shape[:2]
    
    for detection in output:
        if len(detection) >= 6:
            x_center, y_center, width, height, conf, cls_id = detection[:6]
            
            cls_id_int = int(cls_id)
            if conf > conf_threshold and cls_id_int in vehicle_classes:
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
                    # Only detect vehicle classes
                    if cls_id in vehicle_classes:
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
    global stop_threads
    while not stop_threads:
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
# Display window and main loop
# -----------------------------
try:
    cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Vehicle Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
except Exception as e:
    print(f"⚠️ Warning: Could not set up display window (possible Qt/Wayland plugin issue). Display will use default settings: {e}")
    # Proceed, but expect the window to open in default mode or fail to open if X server is not running

last_frames = [None for _ in cameras]

print("🚀 Starting vehicle detection...")

# -----------------------------
# Main loop with GitHub push
# -----------------------------
while True:
    frames = []
    current_time = time.time()
    
    # Process each camera stream
    for i, cam in enumerate(cameras):
        # Get frame from queue (non-blocking)
        try:
            frame = frame_queues[i].get_nowait()
            last_frames[i] = frame.copy()
        except:
            # If queue is empty, use the last good frame
            frame = last_frames[i]

        if frame is None:
            # Display "No Signal" placeholder
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"No Signal - {cam['name']}", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Run Inference
            detections = run_inference(frame)
            
            # Draw bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['conf']
                label = vehicle_classes.get(det['class_id'], "Unknown")
                
                # Draw the box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display camera name
            cv2.putText(frame, cam['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # -----------------------------
            # GitHub Push Logic
            # -----------------------------
            # Push detection results to GitHub if enabled and enough time has passed
            if (ENABLE_GITHUB_PUSH and GITHUB_AVAILABLE and 
                len(detections) >= MIN_DETECTIONS_TO_SAVE and
                (current_time - last_push_time) >= PUSH_INTERVAL_SECONDS):
                
                # Save frame with detections drawn
                frame_to_save = frame.copy() if SAVE_DETECTION_IMAGES else None
                
                # Push in a separate thread to avoid blocking the main loop
                push_thread = threading.Thread(
                    target=push_detection_event,
                    args=(PROJECT_DIR, detections, cam['name'], frame_to_save, True),
                    daemon=True
                )
                push_thread.start()
                last_push_time = current_time

        frames.append(frame)

    # Combine frames and display
    if frames and all(f is not None for f in frames):
        try:
            # Standardize height before combining
            target_h = 480
            resized = [cv2.resize(f, (int(f.shape[1]*target_h/f.shape[0]), target_h)) for f in frames]
            combined = cv2.hconcat(resized)
            cv2.imshow("Vehicle Detection", combined)
        except Exception as e:
            # Catch errors if frames are corrupted or resizing fails
            # This is typically where the original SegFault was happening during CPU fallback,
            # but is now less likely on the Hailo path.
            print(f"⚠️ Display error (frame processing failed): {e}")


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup with final push
# -----------------------------
stop_threads = True

# Final push of any remaining results
if ENABLE_GITHUB_PUSH and GITHUB_AVAILABLE:
    print("📤 Final push to GitHub...")
    push_to_github(PROJECT_DIR, "Final detection results - session ended")

for cap in caps:
    if cap.isOpened():
        cap.release()
cv2.destroyAllWindows()

if HAILO_AVAILABLE and target:
    try:
        target.release()
    except Exception as e:
        print(f"⚠️ Failed to release Hailo VDevice: {e}")
        
print("👋 Cleanup complete")
