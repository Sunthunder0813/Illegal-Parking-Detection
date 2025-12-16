#!/usr/bin/env python3
# Raspberry Pi 5 + Hailo-8L
# Vehicle detection with multi-camera support

import cv2
import numpy as np
import threading
from queue import Queue
import sys
import time

# -----------------------------
# Check Hailo
# -----------------------------
try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
    HAILO_AVAILABLE = True
    print("✅ Hailo platform detected")
except ImportError:
    HAILO_AVAILABLE = False
    print("⚠️ Hailo not available, using CPU fallback")
    from ultralytics import YOLO

# -----------------------------
# Model setup
# -----------------------------
if HAILO_AVAILABLE:
    HEF_PATH = "yolov8n.hef"  # <-- Replace with your compiled HEF
    try:
        hef = HEF(HEF_PATH)
    except Exception as e:
        print(f"❌ Failed to load HEF: {e}")
        sys.exit(1)

    params = VDevice.create_params()
    target = VDevice(params)
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
else:
    model = YOLO("yolov8n.pt")
    target = None

# -----------------------------
# Vehicle classes
# -----------------------------
vehicle_classes = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# -----------------------------
# Camera info
# -----------------------------
username = "admin"
password = ""
cameras = [
    {"ip": "192.168.18.2", "name": "Camera 1"},
    {"ip": "192.168.18.71", "name": "Camera 2"}
]

frame_queues = [Queue(maxsize=2) for _ in cameras]
stop_threads = False

# -----------------------------
# Hailo preprocessing
# -----------------------------
INPUT_HEIGHT = 640
INPUT_WIDTH = 640

def preprocess_frame(frame):
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized, axis=0)
    return input_data

def postprocess_results(output, frame_shape, conf_threshold=0.5):
    detections = []
    h, w = frame_shape[:2]

    for detection in output:
        if len(detection) >= 6:
            x_center, y_center, width, height, conf, cls_id = detection[:6]
            if conf > conf_threshold and int(cls_id) in vehicle_classes:
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'class_id': int(cls_id)
                })
    return detections

def run_hailo_inference(frame):
    input_data = preprocess_frame(frame)
    with InferVStreams(network_group, network_group_params) as infer_pipeline:
        input_dict = {network_group.get_input_vstream_infos()[0].name: input_data}
        output = infer_pipeline.infer(input_dict)
    output_name = list(output.keys())[0]
    raw_output = output[output_name][0]
    return postprocess_results(raw_output, frame.shape)

def run_inference(frame):
    if HAILO_AVAILABLE:
        return run_hailo_inference(frame)
    else:
        results = model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': float(box.conf[0]),
                        'class_id': cls_id
                    })
        return detections

# -----------------------------
# Camera threads
# -----------------------------
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

caps = []
threads = []
for i, cam in enumerate(cameras):
    rtsp_url = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"❌ Cannot connect to {cam['name']}")
    else:
        print(f"✅ Connected to {cam['name']}")
        t = threading.Thread(target=camera_reader, args=(cap, frame_queues[i]), daemon=True)
        t.start()
        threads.append(t)
    caps.append(cap)

# -----------------------------
# Display window
# -----------------------------
cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Vehicle Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
last_frames = [None for _ in cameras]

print("🚀 Starting vehicle detection...")

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
            cv2.putText(frame, f"No Signal - {cam['name']}", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            detections = run_inference(frame)
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['conf']
                label = vehicle_classes[det['class_id']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, cam['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frames.append(frame)

    # Combine frames horizontally
    target_h = 720
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        resized.append(cv2.resize(f, (int(w*target_h/h), target_h)))
    combined = cv2.hconcat(resized)
    cv2.imshow("Vehicle Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stop_threads = True
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
if target:
    target.release()
print("👋 Cleanup complete")
