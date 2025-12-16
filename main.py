import cv2
from ultralytics import YOLO
import threading
from queue import Queue

# Load YOLO model
model = YOLO("yolov8n.pt")

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
cv2.namedWindow("IP Camera Stream", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("IP Camera Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Store last valid frames
last_frames = [None for _ in cameras]

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
            frame = cv2.zeros((480, 640, 3), dtype='uint8')
            cv2.putText(frame, f"No Signal - {cameras[i]['name']}", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Filter and draw vehicles (car, motorcycle, e-bike)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = vehicle_classes[cls_id]
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
    cv2.imshow("IP Camera Stream", combined)
    

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_threads = True
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
