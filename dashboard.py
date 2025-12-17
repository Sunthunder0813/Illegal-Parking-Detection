import os
import random
import cv2
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, Response

# Import detection models
from models import HailoDetector, Detection, InferenceResult, VEHICLE_CLASSES, ALL_DETECTION_CLASSES

# Add PERSON_CLASSES and ALL_DETECTION_CLASSES for broader detection
PERSON_CLASSES = ['person']
ALL_DETECTION_CLASSES = VEHICLE_CLASSES + PERSON_CLASSES

# Initialize Flask app
app = Flask(__name__, 
    static_folder="dashboard_static", 
    template_folder="dashboard_templates"
)

# Camera Configuration
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = ""
CAMERAS = [
    {"ip": "192.168.18.2", "name": "Camera 1", "id": 0},
    {"ip": "192.168.18.71", "name": "Camera 2", "id": 1}
]

# Reconnection settings - ULTRA FAST recovery
MAX_RETRY_ATTEMPTS = 2  # Minimal retries for faster cycle
RETRY_DELAY_SECONDS = 0.1  # 100ms between retries
RECONNECT_INTERVAL = 0.3   # 300ms before next retry cycle
FRAME_TIMEOUT_SECONDS = 3  # Increased to 3 seconds for stability
CONNECTION_TIMEOUT_MS = 1500  # 1.5 second connection timeout
READ_TIMEOUT_MS = 1000  # 1 second read timeout

# Camera stream objects
camera_streams = {}

# Initialize the detector (Hailo or CPU fallback)
detector = None

def init_detector():
    """Initialize the vehicle detector"""
    global detector
    try:
        detector = HailoDetector()
        print(f"✅ Detector initialized: {detector.device_type}")
    except Exception as e:
        print(f"⚠️ Failed to initialize detector: {e}")
        detector = None

class CameraStream:
    def __init__(self, camera_config):
        self.camera = camera_config
        self.frame = None
        self.running = False
        self.connected = False
        self.lock = threading.Lock()
        self.cap = None
        self.retry_count = 0
        self.last_frame_time = None
        self.status_message = "Initializing..."
        self.frame_timeout = False
        self.consecutive_failures = 0
        self.last_connect_attempt = None
        self.last_connect_success = None
        self.stabilization_period = 5  # Increased to 5 seconds
        self.valid_frame_count = 0  # Track consecutive valid frames
        self.frames_needed_for_stable = 10  # Need 10 valid frames to be "stable"
        self.is_stabilizing = True  # Flag for stabilization mode
        self.reconnect_cycles = 0  # Track consecutive reconnect cycles
        
        # Detection-related attributes
        self.detections = []  # Current detections
        self.detection_lock = threading.Lock()
        self.last_detection_time = None
        self.detection_enabled = True  # Enable/disable detection
        self.detection_interval = 0.1  # Run detection every 100ms
        self.last_detection_run = None
        self.detect_all_classes = True  # Enable detection of all objects (persons, vehicles, etc.)
        
    def get_rtsp_url(self):
        """Generate RTSP URL for the camera"""
        ip = self.camera['ip']
        if CAMERA_PASSWORD:
            return f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{ip}:554/stream1"
        else:
            return f"rtsp://{CAMERA_USERNAME}@{ip}:554/stream1"
    
    def run_detection(self, frame):
        """Run object detection on a frame"""
        global detector
        if detector is None or not self.detection_enabled:
            return []
        
        try:
            # Check if enough time has passed since last detection
            now = datetime.now()
            if self.last_detection_run:
                elapsed = (now - self.last_detection_run).total_seconds()
                if elapsed < self.detection_interval:
                    return self.detections  # Return cached detections
            
            self.last_detection_run = now
            
            # Run detection - detect all objects including persons
            if self.detect_all_classes:
                # Use detect_all to get persons, vehicles, and other objects
                detections = detector.detect_all(frame, recognize_plates=False)
            else:
                # Only detect vehicles with plate recognition
                detections = detector.detect(frame, recognize_plates=True, vehicle_only=True)
            
            with self.detection_lock:
                self.detections = detections
                self.last_detection_time = now
            
            return detections
        except Exception as e:
            print(f"❌ [{self.camera['name']}] Detection error: {e}")
            return []
    
    def get_detections(self):
        """Get current detections"""
        with self.detection_lock:
            return list(self.detections)
    
    def draw_detections(self, frame):
        """Draw detection boxes on frame"""
        if frame is None:
            return frame
        
        detections = self.get_detections()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Different colors for different object types
            label_lower = det.label.lower()
            if label_lower == 'person':
                color = (255, 0, 255)  # Magenta for persons
            elif label_lower in [v.lower() for v in VEHICLE_CLASSES]:
                color = (0, 255, 0)  # Green for vehicles
            else:
                color = (0, 255, 255)  # Yellow for other objects
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.label}: {det.confidence:.2f}"
            if hasattr(det, 'plate_number') and det.plate_number:
                label += f" | {det.plate_number}"
            
            # Background for label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw plate bbox if available
            if hasattr(det, 'plate_bbox') and det.plate_bbox:
                px1, py1, px2, py2 = det.plate_bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
        
        return frame
    
    def start(self):
        """Start camera stream"""
        if self.running:
            return
        self.running = True
        self.status_message = "Starting..."
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()
        
    def stop(self):
        """Stop camera stream"""
        self.running = False
        self.connected = False
        if self.cap:
            self.cap.release()
    
    def trigger_reconnect(self):
        """Force an immediate reconnection attempt"""
        self.connected = False
        self.frame_timeout = False
        self.status_message = "Reconnecting..."
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
    
    def _try_connect(self):
        """Attempt to connect to camera - ULTRA FAST"""
        rtsp_url = self.get_rtsp_url()
        self.last_connect_attempt = datetime.now()
        
        while self.running and self.retry_count < MAX_RETRY_ATTEMPTS:
            self.retry_count += 1
            self.status_message = f"Connecting... ({self.retry_count}/{MAX_RETRY_ATTEMPTS})"
            print(f"📹 [{self.camera['name']}] {self.status_message}")
            
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
            
            # Ultra-fast connection settings
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CONNECTION_TIMEOUT_MS)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if self.cap.isOpened():
                ret = self.cap.grab()
                if ret:
                    ret, _ = self.cap.retrieve()
                    if ret:
                        self.connected = True
                        self.retry_count = 0
                        self.frame_timeout = False
                        self.consecutive_failures = 0
                        self.last_connect_success = datetime.now()
                        self.last_frame_time = datetime.now()
                        self.valid_frame_count = 0  # Reset valid frame counter
                        self.is_stabilizing = True  # Enter stabilization mode
                        # Extend stabilization based on reconnect cycles
                        self.stabilization_period = min(1 + (self.reconnect_cycles * 1), 5)
                        self.status_message = "Stabilizing..."
                        print(f"✅ [{self.camera['name']}] Connected! Stabilizing for {self.stabilization_period}s...")
                        return True
            
            time.sleep(RETRY_DELAY_SECONDS)
        
        self.connected = False
        self.status_message = "Offline - Retrying..."
        return False
            
    def _capture_loop(self):
        """Continuous capture loop - ULTRA FAST reconnection"""
        while self.running:
            # Check frame timeout only if connected AND stable (not in stabilization mode)
            if self.connected and self.last_frame_time and self.last_connect_success:
                time_since_connect = (datetime.now() - self.last_connect_success).total_seconds()
                time_since_frame = (datetime.now() - self.last_frame_time).total_seconds()
                
                # During stabilization, use much longer timeout
                if self.is_stabilizing:
                    effective_timeout = max(FRAME_TIMEOUT_SECONDS * 1, 10)  # At least 10 seconds during stabilization
                else:
                    effective_timeout = FRAME_TIMEOUT_SECONDS
                
                # Only check timeout after stabilization period AND if we've had enough valid frames
                if time_since_connect > self.stabilization_period and self.valid_frame_count >= self.frames_needed_for_stable:
                    self.is_stabilizing = False
                    self.reconnect_cycles = 0  # Reset cycles after successful stabilization
                    
                if not self.is_stabilizing and time_since_frame > effective_timeout:
                    print(f"⏱️ [{self.camera['name']}] Frame timeout ({time_since_frame:.1f}s) - Reconnecting...")
                    self.reconnect_cycles += 1
                    self._instant_reconnect()
                    continue
                elif self.is_stabilizing and time_since_frame > effective_timeout:
                    # Even during stabilization, if we haven't received anything for 10+ seconds, reconnect
                    print(f"⏱️ [{self.camera['name']}] Stabilization timeout ({time_since_frame:.1f}s) - Reconnecting...")
                    self.reconnect_cycles += 1
                    self._instant_reconnect()
                    continue
            
            if not self.connected:
                if not self._try_connect():
                    time.sleep(RECONNECT_INTERVAL)
                    self.retry_count = 0
                    continue
            
            try:
                if self.cap and self.cap.grab():
                    ret, frame = self.cap.retrieve()
                    if ret:
                        frame = cv2.resize(frame, (640, 480))
                        
                        # Run detection on frame
                        if self.detection_enabled and not self.is_stabilizing:
                            self.run_detection(frame)
                        
                        with self.lock:
                            self.frame = frame
                            self.last_frame_time = datetime.now()
                            self.frame_timeout = False
                        
                        # Track valid frames for stabilization
                        self.valid_frame_count += 1
                        if self.is_stabilizing and self.valid_frame_count >= self.frames_needed_for_stable:
                            time_since_connect = (datetime.now() - self.last_connect_success).total_seconds()
                            if time_since_connect > self.stabilization_period:
                                self.is_stabilizing = False
                                self.reconnect_cycles = 0
                                self.status_message = "Connected"
                                print(f"🎯 [{self.camera['name']}] Stabilized after {self.valid_frame_count} frames!")
                        elif self.is_stabilizing:
                            self.status_message = f"Stabilizing... ({self.valid_frame_count}/{self.frames_needed_for_stable})"
                        else:
                            self.status_message = "Connected"
                        
                        self.consecutive_failures = 0
                    else:
                        self._handle_read_failure()
                else:
                    self._handle_read_failure()
            except Exception as e:
                print(f"❌ [{self.camera['name']}] Error: {e}")
                self._handle_read_failure()
            
            time.sleep(0.015)  # ~66 FPS capture rate
        
        if self.cap:
            self.cap.release()
    
    def _handle_read_failure(self):
        """Handle read failure - with stabilization awareness"""
        self.consecutive_failures += 1
        
        # During stabilization, be more lenient (allow more failures)
        failure_threshold = 10 if self.is_stabilizing else 3
        
        if self.consecutive_failures >= failure_threshold:
            print(f"⚠️ [{self.camera['name']}] {self.consecutive_failures} failures - Reconnecting...")
            self.reconnect_cycles += 1
            self._instant_reconnect()
    
    def _instant_reconnect(self):
        """Instant reconnection - NO DELAYS"""
        self.connected = False
        self.frame_timeout = False
        self.consecutive_failures = 0
        self.retry_count = 0
        self.valid_frame_count = 0
        self.is_stabilizing = True
        self.status_message = "Reconnecting..."
        self.last_connect_success = None
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
    
    def get_frame(self):
        """Get current frame as JPEG with detections drawn"""
        with self.lock:
            if self.frame is None or not self.connected:
                placeholder = self._create_placeholder()
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            
            # Draw detections on frame copy
            frame_with_detections = self.draw_detections(self.frame.copy())
            _, jpeg = cv2.imencode('.jpg', frame_with_detections)
            return jpeg.tobytes()
    
    def get_status(self):
        """Get camera status info - READ ONLY, no state modification"""
        frame_timed_out = False
        seconds_since = None
        
        if self.last_frame_time:
            seconds_since = (datetime.now() - self.last_frame_time).total_seconds()
            
            # Only report timeout if NOT stabilizing and past stabilization period
            if not self.is_stabilizing and self.last_connect_success:
                time_since_connect = (datetime.now() - self.last_connect_success).total_seconds()
                if time_since_connect > self.stabilization_period:
                    frame_timed_out = seconds_since > FRAME_TIMEOUT_SECONDS
            elif not self.last_connect_success:
                frame_timed_out = seconds_since > FRAME_TIMEOUT_SECONDS
        
        # Determine display status
        if self.is_stabilizing and self.connected:
            display_status = f"Stabilizing... ({self.valid_frame_count}/{self.frames_needed_for_stable})"
        elif frame_timed_out:
            display_status = 'Reconnecting...'
        else:
            display_status = self.status_message
        
        return {
            'id': self.camera['id'],
            'name': self.camera['name'],
            'ip': self.camera['ip'],
            'connected': self.connected and not frame_timed_out,
            'status_message': display_status,
            'retry_count': self.retry_count,
            'last_frame': self.last_frame_time.strftime('%H:%M:%S') if self.last_frame_time else None,
            'frame_timeout': frame_timed_out,
            'seconds_since_frame': round(seconds_since, 1) if seconds_since else None,
            'stabilizing': self.is_stabilizing,
            'detection_count': len(self.detections),
            'detections': [d.to_dict() for d in self.get_detections()]
        }
    
    def _create_placeholder(self):
        """Create a placeholder image when camera is not available"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2.putText(img, self.camera['name'], (220, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(img, self.status_message, (150, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        cv2.putText(img, f"IP: {self.camera['ip']}", (230, 290), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return img

def init_cameras():
    """Initialize all camera streams and detector"""
    # Initialize the detector first
    init_detector()
    
    # Initialize camera streams
    for cam in CAMERAS:
        camera_streams[cam['id']] = CameraStream(cam)
        camera_streams[cam['id']].start()

# Mock violation data
class MockViolation:
    def __init__(self, id, timestamp, location, status, duration):
        self.id = id
        self.timestamp = timestamp
        self.location = location
        self.status = status
        self.duration = duration

def generate_mock_violations():
    """Generate sample violation data"""
    locations = [
        "Main Street - Block A",
        "Parking Lot B - Entrance",
        "Highway 101 - Exit 5",
        "Downtown Plaza",
        "Mall Parking Area",
        "School Zone - North"
    ]
    
    violations = []
    now = datetime.now()
    random.seed(now.minute)
    
    for i in range(random.randint(5, 10)):
        violations.append(MockViolation(
            id=i + 1,
            timestamp=now - timedelta(minutes=random.randint(1, 480)),
            location=random.choice(locations),
            status=random.choice(['active', 'active', 'resolved']),
            duration=random.uniform(5, 120)
        ))
    
    return sorted(violations, key=lambda x: x.timestamp, reverse=True)

def get_violation_stats(violations):
    """Calculate violation statistics from mock data"""
    total = len(violations)
    active = sum(1 for v in violations if v.status == 'active')
    resolved = total - active
    avg_duration = sum(v.duration for v in violations) / total if total > 0 else 0
    
    return {
        'total': total,
        'active': active,
        'resolved': resolved,
        'avg_duration': round(avg_duration, 1)
    }

# ============== ROUTES ==============

@app.route("/")
def dashboard():
    """Main dashboard route"""
    violations = generate_mock_violations()
    stats = get_violation_stats(violations)
    
    return render_template(
        "dashboard.html",
        total=stats['total'],
        active=stats['active'],
        resolved=stats['resolved'],
        avg_duration=stats['avg_duration'],
        recent=violations[:5],
        last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        cameras=CAMERAS  # Pass cameras list to template
    )

@app.route("/api/stats")
def api_stats():
    """API endpoint for real-time stats updates"""
    violations = generate_mock_violations()
    stats = get_violation_stats(violations)
    recent = [
        {
            'id': v.id,
            'timestamp': v.timestamp.strftime('%H:%M:%S'),
            'location': v.location,
            'status': v.status,
            'duration': round(v.duration, 1)
        }
        for v in violations[:5]
    ]
    
    return jsonify({
        **stats,
        'recent': recent,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route("/api/cameras")
def api_cameras():
    """Get detailed camera status - called by frontend every 2 seconds"""
    status = []
    for cam in CAMERAS:
        stream = camera_streams.get(cam['id'])
        if stream:
            status.append(stream.get_status())
        else:
            status.append({
                'id': cam['id'],
                'name': cam['name'],
                'ip': cam['ip'],
                'connected': False,
                'status_message': 'Not initialized',
                'retry_count': 0,
                'last_frame': None
            })
    return jsonify(status)

@app.route("/api/server/status")
def api_server_status():
    """Server status endpoint for connection checking"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    """Video streaming route - optimized for fast delivery"""
    def generate():
        while True:
            stream = camera_streams.get(camera_id)
            if stream:
                frame = stream.get_frame()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.02)  # 50 FPS max
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/health")
def health():
    """Health check endpoint"""
    camera_status = all(s.connected for s in camera_streams.values()) if camera_streams else False
    detector_status = detector is not None
    return jsonify({
        "status": "healthy", 
        "mode": "demo",
        "cameras_online": camera_status,
        "detector_initialized": detector_status,
        "detector_type": detector.device_type if detector else None,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/api/detections")
def api_detections():
    """Get all current detections from all cameras"""
    all_detections = []
    for cam_id, stream in camera_streams.items():
        detections = stream.get_detections()
        for det in detections:
            det_dict = det.to_dict()
            det_dict['camera_id'] = cam_id
            det_dict['camera_name'] = stream.camera['name']
            all_detections.append(det_dict)
    
    return jsonify({
        'detections': all_detections,
        'total_count': len(all_detections),
        'timestamp': datetime.now().isoformat()
    })

@app.route("/api/detections/<int:camera_id>")
def api_camera_detections(camera_id):
    """Get detections for a specific camera"""
    stream = camera_streams.get(camera_id)
    if stream:
        detections = stream.get_detections()
        return jsonify({
            'camera_id': camera_id,
            'camera_name': stream.camera['name'],
            'detections': [d.to_dict() for d in detections],
            'count': len(detections),
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({
        'error': 'Camera not found'
    }, 404)

@app.route("/api/camera/<int:camera_id>/reconnect", methods=['POST'])
def api_camera_reconnect(camera_id):
    """Force reconnect a specific camera"""
    stream = camera_streams.get(camera_id)
    if stream:
        stream.trigger_reconnect()
        return jsonify({
            'success': True,
            'message': f'Reconnecting camera {camera_id}...'
        })
    return jsonify({
        'success': False,
        'message': 'Camera not found'
    }, 404)

@app.route("/api/cameras/reconnect-all", methods=['POST'])
def api_cameras_reconnect_all():
    """Force reconnect all cameras"""
    for stream in camera_streams.values():
        stream.trigger_reconnect()
    return jsonify({
        'success': True,
        'message': 'Reconnecting all cameras...'
    })

# ============== MAIN ==============

if __name__ == "__main__":
    print("🎥 Initializing cameras and detector...")
    init_cameras()
    
    if detector:
        print(f"🤖 Detector ready: {detector.device_type}")
    else:
        print("⚠️ Detector not available - running without detection")
    
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Starting dashboard on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

