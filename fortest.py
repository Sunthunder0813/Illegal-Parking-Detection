#!/usr/bin/env python3
# Raspberry Pi 5 + Hailo AI Accelerator
# Multi-camera vehicle detection (YOLOv8)

import cv2
import numpy as np
import threading
from queue import Queue
import sys
import os
import time

# ===============================
# Hailo / Fallback Detection
# ===============================
try:
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams
    )
    HAILO_AVAILABLE = True
    print("✅ Hailo platform detected")
except ImportError:
    HAILO_AVAILABLE = False
    print("⚠️ Hailo not available – using Ultralytics fallback")
    from ultralytics import YOLO


# ===============================
# Model Setup
# ===============================
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

if HAILO_AVAILABLE:
    HEF_PATH = "/home/set-admin/Illegal-Parking-Detection/yolov8n.hef"

    if not os.path.exists(HEF_PATH):
        print(f"❌ HEF file not found: {HEF_PATH}")
        sys.exit(1)

    hef = HEF(HEF_PATH)

    params = VDevice.create_params()
    target = VDevice(params)

    configure_params = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe
    )

    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    # ✅ Create inference pipeline ONCE
    infer_pipeline = InferVStreams(network_group, network_group_params)

else:
    model = YOLO("yolov8n.pt")
    infer_pipeline = None
    target = None


# ===============================
# Vehicle Classes (COCO IDs)
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
# Preprocess for Hailo
# ===============================
def preprocess_frame(frame):
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0).astype(np.uint8)


# ===============================
# Postprocess YOLOv8 HEF output
# ===============================
def postprocess_results(hailo_output, frame_shape, conf_threshold=0.5):
    detections = []
    h, w = frame_shape[:2]

    boxes = hailo_output["boxes"]
    scores = hailo_output["scores"]
    classes = hailo_output["classes"]
    num_dets = int(hailo_output["num_detections"][0])

    for i in range(num_dets):
        conf = float(scores[i])
        cls_id = int(classes[i])

        if conf < conf_threshold or cls_id not in vehicle_classes:
            continue

        y1, x1, y2, x2 = boxes[i]

        x1 = int(x1 * w)
        x2 = int(x2 * w)
        y1 = int(y1 * h)
        y2 = int(y2 * h)

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "class_id": cls_id
        })

    return detections


# ===============================
# Inference Functions
# ===============================
def run_hailo_inference(frame):
    input_data = preprocess_frame(frame)
    input_name = network_group.get_input_vstream_infos()[0].name
    output = infer_pipeline.infer({input_name: input_data})
    return postprocess_results(output, frame.shape)


def run_inference(frame):
    if HAILO_AVAILABLE:
        return run_hailo_inference(frame)
    else:
        results = model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(box.conf[0]),
                        "class_id": cls_id
                    })
        return detections


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


# ===============================
# Open Cameras
# ===============================
caps = []
threads = []

for i, cam in enumerate(cameras):
    rtsp = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"❌ Cannot connect to {cam['name']}")
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
cv2.namedWindow("Hailo Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Hailo Vehicle Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

last_frames = [None] * len(cameras)

print("🚀 Starting vehicle detection...")


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
            detections = run_inference(frame)

            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                label = vehicle_classes[d["class_id"]]
                conf = d["conf"]

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

    # Combine frames
    target_h = 720
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        resized.append(cv2.resize(f, (int(w * target_h / h), target_h)))

    combined = cv2.hconcat(resized)
    cv2.imshow("Hailo Vehicle Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ===============================
# Cleanup
# ===============================
stop_threads = True

for cap in caps:
    cap.release()

if infer_pipeline:
    infer_pipeline.release()

if target:
    target.release()

cv2.destroyAllWindows()
print("👋 Shutdown complete")
