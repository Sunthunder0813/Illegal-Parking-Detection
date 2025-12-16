#!/usr/bin/env python3
# Raspberry Pi 5 - YOLOv8 CPU Fallback
# Multi-camera vehicle detection (NO Hailo HEF required)

import cv2
import numpy as np
import threading
from queue import Queue
import time
import sys

from ultralytics import YOLO

print("⚠️ Running CPU fallback (Hailo HEF not present)")

# ===============================
# Load YOLOv8 model
# ===============================
MODEL_PATH = "yolov8n.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# ===============================
# Vehicle Classes (COCO)
# ===============================
vehicle_classes = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ===============================
# Camera Configuration
# ===============================
username = "admin"
password = ""
cameras = [
    {"ip": "192.168.18.2", "name": "Camera 1"},
    {"ip": "192.168.18.71", "name": "Camera 2"}
]

frame_queues = [Queue(maxsize=2) for _ in cameras]
stop_threads = False


# ===============================
# Camera Reader Thread
# ===============================
def camera_reader(cap, queue):
    global stop_threads
    while not stop_threads:
        ret, frame = cap.read()
        if ret:
            if queue.full():
                try:
                    queue.get_nowait()
                except:
                    pass
            queue.put(frame)
        else:
            time.sleep(0.05)


# ===============================
# Open Cameras
# ===============================
caps = []
threads = []

for i, cam in enumerate(cameras):
    rtsp = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 25)

    if not cap.isOpened():
        print(f"❌ Cannot connect to {cam['name']} ({cam['ip']})")
    else:
        print(f"✅ Connected to {cam['name']}")
        t = threading.Thread(
            target=camera_reader,
            args=(cap, frame_queues[i]),
            daemon=True
        )
        t.start()
        threads.append(t)

    caps.append(cap)


# ===============================
# Display Window
# ===============================
cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Vehicle Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

last_frames = [None] * len(cameras)

print("🚀 Starting vehicle detection (CPU)...")

# ===============================
# Main Loop
# ===============================
while True:
    frames = []

    for i, cam in enumerate(cameras):
        try:
            frame = frame_queues[i].get_nowait()
            last_frames[i] = frame.copy()
        except:
            frame = last_frames[i]

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"No Signal - {cam['name']}",
                (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        else:
            # Run YOLO inference
            results = model(frame, imgsz=640, conf=0.5, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in vehicle_classes:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = vehicle_classes[cls_id]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            cv2.putText(
                frame,
                cam["name"],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        frames.append(frame)

    # Combine frames horizontally
    target_h = 720
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        resized.append(cv2.resize(f, (int(w * target_h / h), target_h)))

    combined = cv2.hconcat(resized)
    cv2.imshow("Vehicle Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ===============================
# Cleanup
# ===============================
stop_threads = True

for cap in caps:
    cap.release()

cv2.destroyAllWindows()
print("👋 Shutdown complete")
