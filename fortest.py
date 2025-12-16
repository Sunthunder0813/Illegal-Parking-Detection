# For Beta Test - Raspberry Pi 5 with Hailo AI Accelerator

import cv2
import numpy as np
import threading
from queue import Queue
import sys

# Check if running on Raspberry Pi with Hailo
try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
    HAILO_AVAILABLE = True
    print("✅ Hailo platform detected")
except ImportError:
    HAILO_AVAILABLE = False
    print("⚠️ Hailo not available - using fallback (ultralytics)")
    from ultralytics import YOLO

# Model setup
if HAILO_AVAILABLE:
    HEF_PATH = "yolov8n.hef"
    hef = HEF(HEF_PATH)
    params = VDevice.create_params()
    target = VDevice(params)
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
else:
    model = YOLO("yolov8n.pt")
    target = None

# Vehicle classes to detect (COCO dataset)
vehicle_classes = {
    1: "E-Bike",      # bicycle
    2: "Car",         # car
    3: "Motorcycle"   # motorcycle
}

# Camera info
username = "admin"
password = ""
cameras = [
    {"ip": "192.168.18.2", "name": "Camera 1"},
    {"ip": "192.168.18.71", "name": "Camera 2"}
]

# Frame queues for each camera
frame_queues = [Queue(maxsize=2) for _ in cameras]
stop_threads = False

# Model input size (adjust based on your HEF model)
INPUT_HEIGHT = 640
INPUT_WIDTH = 640

def preprocess_frame(frame):
    """Preprocess frame for Hailo inference"""
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize to 0-1 range (if required by model)
    normalized = rgb.astype(np.float32) / 255.0
    # Add batch dimension
    input_data = np.expand_dims(normalized, axis=0)
    return input_data

def postprocess_results(output, frame_shape, conf_threshold=0.5):
    """Post-process Hailo output to get bounding boxes"""
    detections = []
    h, w = frame_shape[:2]
    
    # Parse output based on YOLOv8 output format
    # Adjust based on your specific HEF model output structure
    for detection in output:
        if len(detection) >= 6:
            x_center, y_center, width, height, conf, cls_id = detection[:6]
            
            if conf > conf_threshold and int(cls_id) in vehicle_classes:
                # Convert to pixel coordinates
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
    """Run inference using Hailo accelerator"""
    input_data = preprocess_frame(frame)
    
    with InferVStreams(network_group, network_group_params) as infer_pipeline:
        input_dict = {network_group.get_input_vstream_infos()[0].name: input_data}
        output = infer_pipeline.infer(input_dict)
    
    # Get output from the first output layer
    output_name = list(output.keys())[0]
    raw_output = output[output_name][0]  # Remove batch dimension
    
    return postprocess_results(raw_output, frame.shape)

def run_inference(frame):
    """Run inference using Hailo or fallback to YOLO"""
    if HAILO_AVAILABLE:
        return run_hailo_inference(frame)
    else:
        # Fallback to ultralytics YOLO
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

def camera_reader(cap, queue, cam_index):
    """Thread function to continuously read frames from camera"""
    global stop_threads
    while not stop_threads:
        ret, frame = cap.read()
        if ret:
            if queue.full():
                try:
                    queue.get_nowait()  # Remove old frame
                except:
                    pass
            queue.put(frame)

# Connect to cameras with optimized settings
caps = []
threads = []
for i, cam in enumerate(cameras):
    rtsp_url = f"rtsp://{username}:{password}@{cam['ip']}:554/h264"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Optimize capture settings for smoother streaming
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    
    if not cap.isOpened():
        print(f"❌ Cannot connect to {cam['name']} ({cam['ip']})")
    else:
        print(f"✅ Connected to {cam['name']} ({cam['ip']})")
        # Start reader thread for this camera
        t = threading.Thread(target=camera_reader, args=(cap, frame_queues[i], i), daemon=True)
        t.start()
        threads.append(t)
    caps.append(cap)

# Create fullscreen window
cv2.namedWindow("IP Camera Stream - Hailo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("IP Camera Stream - Hailo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Store last valid frames
last_frames = [None for _ in cameras]

print("🚀 Starting Hailo-accelerated vehicle detection...")

# Display live video
while True:
    frames = []
    
    for i, cam in enumerate(cameras):
        frame = None
        
        # Get latest frame from queue
        try:
            frame = frame_queues[i].get_nowait()
            last_frames[i] = frame.copy()
        except:
            frame = last_frames[i]  # Use last valid frame
        
        if frame is None:
            # Create blank frame if no frame available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"No Signal - {cameras[i]['name']}", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Run Hailo inference
            detections = run_inference(frame)
            
            # Draw detected vehicles
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['conf']
                label = vehicle_classes[det['class_id']]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add camera label
            cv2.putText(frame, cameras[i]['name'], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    # Resize frames to same height and concatenate horizontally
    height = 720
    resized_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        new_width = int(w * height / h)
        resized = cv2.resize(frame, (new_width, height))
        resized_frames.append(resized)
    
    combined = cv2.hconcat(resized_frames)
    cv2.imshow("IP Camera Stream - Hailo", combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_threads = True
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
# Only release target if Hailo is available
if target:
    target.release()
print("👋 Cleanup complete")
