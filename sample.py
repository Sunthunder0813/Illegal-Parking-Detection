#!/usr/bin/env python3
# Raspberry Pi 5 + Hailo-8L
# Vehicle detection with multi-camera support and optional Hailo GStreamer

import os
import cv2
import numpy as np
import threading
from queue import Queue
import time
import subprocess
import signal

# =============================================================================
# Environment fixes
# =============================================================================
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force X11
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

# =============================================================================
# Graceful shutdown
# =============================================================================
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n⚠️ Shutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# Paths and configuration
# =============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PT_MODEL = os.path.join(PROJECT_DIR, "yolov8n.pt")
ONNX_MODEL = os.path.join(PROJECT_DIR, "yolov8n.onnx")
HEF_MODEL = os.path.join(PROJECT_DIR, "yolov8n.hef")

username = "admin"
password = ""
cameras = [
    {"ip": "192.168.18.2", "name": "Camera 1"},
    {"ip": "192.168.18.71", "name": "Camera 2"}
]
frame_queues = [Queue(maxsize=1) for _ in cameras]
stop_threads = False

# =============================================================================
# GitHub push (optional)
# =============================================================================
ENABLE_GITHUB_PUSH = True
PUSH_INTERVAL_SECONDS = 60
SAVE_DETECTION_IMAGES = True
MIN_DETECTIONS_TO_SAVE = 1
last_push_time = 0

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

# =============================================================================
# Vehicle classes
# =============================================================================
vehicle_classes = {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# =============================================================================
# Hailo detection setup
# =============================================================================
HAILO_AVAILABLE = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    import hailo
    from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams

    HAILO_AVAILABLE = True
    print("✅ Hailo platform detected")
except ImportError as e:
    print(f"⚠️ Hailo platform or GStreamer not available: {e}. CPU fallback will be used.")
    from ultralytics import YOLO

# =============================================================================
# HEF auto-download
# =============================================================================
def download_hef():
    """Download HEF file if missing"""
    if os.path.isfile(HEF_MODEL):
        return True
    url = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8l/yolov8n.hef"
    try:
        subprocess.run(["wget", "-q", "-O", HEF_MODEL, url], timeout=120, check=True)
        if os.path.isfile(HEF_MODEL) and os.path.getsize(HEF_MODEL) > 1000:
            print(f"✅ HEF downloaded: {HEF_MODEL}")
            return True
    except Exception as e:
        print(f"⚠️ Failed to download HEF: {e}")
    return False

if HAILO_AVAILABLE and not os.path.isfile(HEF_MODEL):
    if not download_hef():
        print("❌ HEF not found. Falling back to CPU.")
        HAILO_AVAILABLE = False

# =============================================================================
# Hailo GStreamer callback
# =============================================================================
if HAILO_AVAILABLE:
    class UserAppCallback(app_callback_class):
        def __init__(self):
            super().__init__()

    def hailo_gst_callback(pad, info, user_data):
        user_data.increment()
        string_to_print = f"[Hailo-GStreamer] Frame count: {user_data.get_count()}\n"
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        rois = hailo.get_roi_from_buffer(buffer)
        for detection in rois.get_objects_typed(hailo.HAILO_DETECTION):
            string_to_print += f"Detection: {detection.get_label()} Confidence: {detection.get_confidence():.2f}\n"
        print(string_to_print)
        return Gst.PadProbeReturn.OK

# =============================================================================
# CPU fallback
# =============================================================================
if not HAILO_AVAILABLE:
    print("⚠️ Using CPU inference (Ultralytics YOLO)")
    model = YOLO(PT_MODEL)

def run_inference(frame):
    if HAILO_AVAILABLE:
        # Hailo inference handled by GStreamer callback
        return []
    else:
        # CPU inference
        results = model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({'bbox': (x1, y1, x2, y2), 'conf': float(box.conf[0]), 'class_id': cls_id})
        return detections

# =============================================================================
# Camera threads
# =============================================================================
def camera_reader(cap, queue, cam_name):
    global stop_threads, shutdown_requested
    while not stop_threads and not shutdown_requested:
        try:
            ret, frame = cap.read()
            if ret:
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
    rtsp_url = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if not cap.isOpened():
        print(f"❌ Cannot connect to {cam['name']}")
    else:
        print(f"✅ Connected to {cam['name']}")
        t = threading.Thread(target=camera_reader, args=(cap, frame_queues[i], cam['name']), daemon=True)
        t.start()
        threads.append(t)
    caps.append(cap)

# =============================================================================
# Display setup
# =============================================================================
display_available = False
try:
    cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
    display_available = True
except Exception as e:
    print(f"⚠️ Display not available: {e}")

last_frames = [None for _ in cameras]

# =============================================================================
# Main loop
# =============================================================================
if __name__ == "__main__":
    if HAILO_AVAILABLE:
        # Start Hailo GStreamer pipeline
        user_data = UserAppCallback()
        app = GStreamerDetectionApp(hailo_gst_callback, user_data)
        print("🚀 Running Hailo inference...")
        try:
            app.run()
        except KeyboardInterrupt:
            print("👋 Hailo GStreamer interrupted")
    else:
        print("🚀 Starting CPU multi-camera loop...")
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
                                label = vehicle_classes.get(det['class_id'], "Unknown")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, cam['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            # GitHub push
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
            print("👋 Interrupted by user")
        except Exception as e:
            print(f"❌ Main loop error: {e}")

# =============================================================================
# Cleanup
# =============================================================================
stop_threads = True
shutdown_requested = True
time.sleep(0.5)

for cap in caps:
    try:
        if cap.isOpened():
            cap.release()
    except:
        pass

try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)
except:
    pass

if ENABLE_GITHUB_PUSH and GITHUB_AVAILABLE:
    try:
        push_to_github(PROJECT_DIR, "Final detection results - session ended")
    except Exception as e:
        print(f"⚠️ Final push failed: {e}")

print("👋 Cleanup complete")
