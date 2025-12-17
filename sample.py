#!/usr/bin/env python3
# Vehicle detection with multi-camera support using HAILO TAPPAS and GStreamer callbacks

import os
import cv2
import numpy as np
import threading
from queue import Queue
import sys
import time
import signal

# -----------------------------
# Graceful shutdown handler
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
# HAILO TAPPAS Setup
# -----------------------------
try:
    from hailo_platform import Device
    from hailo_tappas.yolov8.pipeline import Yolov8Pipeline
    from hailo_tappas.common.data import PipelineMode
    print("✅ Hailo TAPPAS detected")
except ImportError:
    print("❌ Hailo TAPPAS not found. Install with: pip install hailo-tappas")
    sys.exit(1)

# -----------------------------
# GStreamer callback import
# -----------------------------
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class, run_pipeline

# -----------------------------
# Paths
# -----------------------------
HEF_MODEL = "/home/set-admin/Illegal-Parking-Detection/yolov8n.hef"
if not os.path.isfile(HEF_MODEL):
    print(f"❌ HEF file not found at {HEF_MODEL}")
    sys.exit(1)

# -----------------------------
# Camera info
# -----------------------------
username = "admin"
password = ""
cameras = [
    {"ip": "192.168.18.2", "name": "Camera 1"},
    {"ip": "192.168.18.71", "name": "Camera 2"}
]
frame_queues = [Queue(maxsize=1) for _ in cameras]

# -----------------------------
# COCO class names (80 classes)
# -----------------------------
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

# Only show these classes
detected_classes = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# -----------------------------
# Hailo GStreamer callback class
# -----------------------------
class HailoFrameCallback(app_callback_class):
    def __init__(self, queue, cam_name):
        super().__init__()
        self.queue = queue
        self.cam_name = cam_name

    def on_new_frame(self, frame):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except:
                pass
        self.queue.put(frame)
        return True  # continue pipeline

# -----------------------------
# Start camera pipelines
# -----------------------------
for i, cam in enumerate(cameras):
    callback = HailoFrameCallback(frame_queues[i], cam['name'])
    gst_command = f"rtspsrc location=rtsp://{username}:{password}@{cam['ip']}:554/h264 ! decodebin ! videoconvert ! appsink"
    run_pipeline(gst_command, callback)
    print(f"✅ Hailo GStreamer pipeline started for {cam['name']}")

# -----------------------------
# Hailo TAPPAS YOLOv8 Pipeline
# -----------------------------
try:
    device = Device()
    pipeline = Yolov8Pipeline(hef_path=HEF_MODEL, device=device, mode=PipelineMode.INFERENCE)
    input_height, input_width = pipeline.input_shape[1:3]
    print(f"✅ Hailo TAPPAS pipeline ready. Input shape: ({input_height}, {input_width})")
except Exception as e:
    print(f"❌ Failed to initialize Hailo TAPPAS pipeline: {e}")
    sys.exit(1)

def run_hailo_tappas_inference(frame):
    resized = cv2.resize(frame, (input_width, input_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
    results = pipeline.infer(input_data)
    detections = []
    for det in results:
        cls_id = int(det['class'])
        if cls_id in detected_classes:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = float(det['score'])
            detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'class_id': cls_id, 'class_name': COCO_CLASS_NAMES[cls_id]})
    return detections

# -----------------------------
# Display setup
# -----------------------------
display_available = False
try:
    cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
    display_available = True
except Exception as e:
    print(f"⚠️ Cannot create display window: {e}")
    print("   Running in headless mode")

last_frames = [None for _ in cameras]

# -----------------------------
# Main loop
# -----------------------------
print("🚀 Starting vehicle detection (HAILO TAPPAS + GStreamer)...")
print("   Press 'q' to quit (or Ctrl+C)")

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
                    detections = run_hailo_tappas_inference(frame)
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        conf = det['conf']
                        label = detected_classes.get(det['class_id'], "Unknown")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, cam['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # GitHub push logic
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

        if display_available and frames and all(f is not None for f in frames):
            try:
                target_h = 480
                resized = [cv2.resize(f, (int(f.shape[1]*target_h/f.shape[0]), target_h)) for f in frames]
                combined = cv2.hconcat(resized)
                cv2.imshow("Vehicle Detection", combined)
            except Exception as e:
                print(f"⚠️ Display error: {e}")
                display_available = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("👋 Quit requested...")
            break

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")
except Exception as e:
    print(f"❌ Main loop error: {e}")

# -----------------------------
# Cleanup
# -----------------------------
print("🧹 Cleaning up...")
shutdown_requested = True
time.sleep(0.5)

try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)
except:
    pass

if ENABLE_GITHUB_PUSH and GITHUB_AVAILABLE:
    print("📤 Final push to GitHub...")
    try:
        push_to_github(PROJECT_DIR, "Final detection results - session ended")
    except Exception as e:
        print(f"⚠️ Final push failed: {e}")

try:
    pipeline.release()
except Exception as e:
    print(f"⚠️ Failed to release Hailo TAPPAS pipeline: {e}")
try:
    device.release()
except Exception as e:
    print(f"⚠️ Failed to release Hailo Device: {e}")

print("👋 Cleanup complete")
