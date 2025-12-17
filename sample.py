#!/usr/bin/env python3
import os
import cv2
import numpy as np
import threading
from queue import Queue
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams

# -----------------------------
# CONFIGURATION
# -----------------------------
HEF_PATH = "/home/set-admin/Illegal-Parking-Detection/yolov8n.hef"
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Camera 1"},
    {"ip": "192.168.18.71", "name": "Camera 2"}
]
USERNAME = "admin"
PASSWORD = ""
CONF_THRESHOLD = 0.3

# COCO classes (we use 0=person, 2=car)
LABELS = {0: "Person", 2: "Car"}

# -----------------------------
# THREADING QUEUES
# -----------------------------
frame_queues = [Queue(maxsize=1) for _ in CAMERAS]
stop_threads = False

# -----------------------------
# CAMERA READER THREAD
# -----------------------------
def camera_reader(cam_index, cam_info):
    global stop_threads
    rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{cam_info['ip']}:554/h264"
    print(f"Connecting to {cam_info['name']} at {rtsp_url}")
    retry_count = 0
    cap = None

    # Retry connection up to 10 times
    while not stop_threads and retry_count < 10:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if cap.isOpened():
            print(f"✅ Connected to {cam_info['name']}")
            break
        else:
            print(f"❌ Cannot connect to {cam_info['name']} ({cam_info['ip']}) [Attempt {retry_count+1}/10]")
            cap.release()
            cap = None
            retry_count += 1
            cv2.waitKey(1000)  # Wait 1 second before retrying

    if cap is None or not cap.isOpened():
        print(f"❌ Failed to connect to {cam_info['name']} after {retry_count} attempts.")
        return

    while not stop_threads:
        ret, frame = cap.read()
        if ret:
            if frame_queues[cam_index].full():
                try: frame_queues[cam_index].get_nowait()
                except: pass
            frame_queues[cam_index].put(frame)
        else:
            print(f"⚠️ Failed to read frame from {cam_info['name']}")
            cv2.waitKey(10)
    cap.release()

# -----------------------------
# LOAD HEF + CONFIGURE DEVICE
# -----------------------------
hef = HEF(HEF_PATH)
vdevice_params = VDevice.create_params()
device = VDevice(vdevice_params)
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_group = device.configure(hef, configure_params)[0]
ng_params = network_group.create_params()

# Input info
input_vstream_info = network_group.get_input_vstream_infos()[0]
INPUT_HEIGHT = input_vstream_info.shape[1]
INPUT_WIDTH = input_vstream_info.shape[2]

# -----------------------------
# PREPROCESS FRAME
# -----------------------------
def preprocess(frame):
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(normalized, axis=0)  # NHWC
    return input_data

# -----------------------------
# POSTPROCESS YOLOV8 OUTPUT
# -----------------------------
def postprocess(output, frame_shape, conf_thresh=0.3):
    # output shape: [N,6] or [N,84*?] depending on HEF
    # Hailo Model Zoo usually outputs [x,y,w,h,conf,class_id]
    detections = []
    h, w = frame_shape[:2]
    for det in output:
        if len(det) < 6:
            continue
        x, y, bw, bh, conf, cls_id = det[:6]
        cls_id = int(cls_id)
        if conf < conf_thresh or cls_id not in LABELS:
            continue
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        detections.append({
            "class_id": cls_id,
            "conf": conf,
            "bbox": (x1, y1, x2, y2)
        })
    return detections

# -----------------------------
# RUN INFERENCE
# -----------------------------
def run_inference(frame):
    input_data = preprocess(frame)
    with InferVStreams(network_group, ng_params) as infer_pipeline:
        input_name = network_group.get_input_vstream_infos()[0].name
        output_dict = infer_pipeline.infer({input_name: input_data})
        output_name = list(output_dict.keys())[0]
        raw_output = output_dict[output_name][0]
        return postprocess(raw_output, frame.shape, CONF_THRESHOLD)

# -----------------------------
# START CAMERA THREADS
# -----------------------------
threads = []
for i, cam in enumerate(CAMERAS):
    t = threading.Thread(target=camera_reader, args=(i, cam), daemon=True)
    t.start()
    threads.append(t)

# -----------------------------
# MAIN DISPLAY LOOP
# -----------------------------
last_frames = [None for _ in CAMERAS]
try:
    while True:
        frames = []
        for i, cam in enumerate(CAMERAS):
            try:
                frame = frame_queues[i].get_nowait()
                last_frames[i] = frame.copy()
            except:
                frame = last_frames[i]

            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"No Signal - {cam['name']}", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                detections = run_inference(frame)
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    conf = det["conf"]
                    label = LABELS[det["class_id"]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, cam["name"], (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            frames.append(frame)

        # Combine frames horizontally
        resized = [cv2.resize(f, (640,480)) for f in frames]
        combined = cv2.hconcat(resized)
        cv2.imshow("YOLOv8 Hailo Detection", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n🛑 Exiting...")
finally:
    stop_threads = True
    for t in threads:
        t.join()
    cv2.destroyAllWindows()
    if device is not None:
        device.release()
