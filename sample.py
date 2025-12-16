#!/usr/bin/env python3
# Raspberry Pi 5 + Hailo-8L
# Vehicle detection with multi-camera support

import os
import cv2
import numpy as np
import threading
from queue import Queue
import sys
import time
import subprocess

# -----------------------------
# Check Hailo Platform
# -----------------------------
HAILO_AVAILABLE = False
try:
    # We only need the hailo_platform library for inference on the Pi
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
    HAILO_AVAILABLE = True
    print("✅ Hailo platform detected")
except ImportError:
    print("⚠️ Hailo platform library (hailo_platform) not found.")
    # Fallback uses Ultralytics YOLO library
    from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
PT_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.pt"
ONNX_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.onnx"
HEF_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.hef"

# -----------------------------
# Export PT -> ONNX if missing
# -----------------------------
if not os.path.isfile(ONNX_MODEL):
    print("⚠️ ONNX model not found, exporting from PT...")
    try:
        from ultralytics import YOLO
        # Exporting with dynamic=True might be better for compatibility, but check Hailo documentation.
        model = YOLO(PT_MODEL)
        model.export(format="onnx", imgsz=640, dynamic=False)
        print(f"✅ Exported ONNX model: {ONNX_MODEL}")
    except Exception as e:
        print(f"❌ Failed to export ONNX: {e}")
        sys.exit(1)

# -----------------------------
# Check for HEF (Model must be pre-compiled on x86 machine)
# -----------------------------
if HAILO_AVAILABLE and not os.path.isfile(HEF_MODEL):
    print("❌ HEF file not found.")
    print("--- CRITICAL ERROR ---")
    print("The Hailo Executable Format (.hef) file must be pre-compiled on an x86 Ubuntu machine.")
    print(f"Please compile {ONNX_MODEL} using the Hailo Dataflow Compiler (DFC) and transfer the resulting HEF file to:")
    print(f"-> {HEF_MODEL}")
    print("Falling back to CPU inference.")
    HAILO_AVAILABLE = False


# -----------------------------
# Model setup
# -----------------------------
if HAILO_AVAILABLE and os.path.isfile(HEF_MODEL):
    try:
        hef = HEF(HEF_MODEL)
        params = VDevice.create_params()
        target = VDevice(params)
        # Assuming PCIe interface for Raspberry Pi 5 AI Kit
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        
        # Determine model input/output info (Crucial for preprocessing)
        input_info = network_group.get_input_vstream_infos()[0]
        global INPUT_HEIGHT, INPUT_WIDTH
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
else:
    # CPU Fallback setup
    print("⚠️ Setting up CPU inference (Ultralytics YOLO).")
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
username = "admin"
password = ""
cameras = [
    {"ip": "192.168.18.2", "name": "Camera 1"},
    {"ip": "192.168.18.71", "name": "Camera 2"}
]

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
    # NOTE: This postprocessing logic is for a generic raw Hailo output (like a transposed YOLO output)
    # The actual HEF output format can vary. You may need to customize this based on the HEF's configuration.
    detections = []
    h, w = frame_shape[:2]
    # The output is expected to be (N, 84, 8400) or similar, then transposed or already flattened.
    # Assuming output is a list/array of [x_center, y_center, width, height, conf, class_id] per box,
    # and coordinates are normalized (0-1).
    
    # If the output is the raw transposed YOLO (84 classes, N boxes), it needs proper handling.
    # Assuming the output tensor is already post-processed or is formatted for easy parsing:
    
    for detection in output:
        # Check if the detection array has at least 6 elements (x, y, w, h, conf, cls)
        if len(detection) >= 6:
            # Scale coordinates and check confidence
            x_center, y_center, width, height, conf, cls_id = detection[:6]
            
            # The class_id from the raw tensor might be a float index, convert and check against known classes
            cls_id_int = int(cls_id)
            if conf > conf_threshold and cls_id_int in vehicle_classes:
                # Convert normalized (0-1) box center/width/height to absolute pixel values
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

while True:
    frames = []
    
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
# Cleanup
# -----------------------------
stop_threads = True
for cap in caps:
    if cap.isOpened():
        cap.release()
cv2.destroyAllWindows()

if HAILO_AVAILABLE and target:
    try:
        # Release the Hailo resources
        target.release()
    except Exception as e:
        print(f"⚠️ Failed to release Hailo VDevice: {e}")
        
print("👋 Cleanup complete")
