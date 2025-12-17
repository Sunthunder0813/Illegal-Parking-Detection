import os
import random
import cv2
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, Response

# Import the detector
try:
    from models import HailoDetector, Detection, VEHICLE_CLASSES, ALL_TARGET_CLASSES
    DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import detector: {e}")
    DETECTOR_AVAILABLE = False

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

# Global detector instance
detector = None
detector_lock = threading.Lock()

def init_detector():
    """Initialize the object detector"""
    global detector
    if not DETECTOR_AVAILABLE:
        print("⚠️ Detector not available - running without detection")
        return False
    
    try:
        print("🔧 Initializing object detector...")
        detector = HailoDetector(
            detect_all_objects=True,  # Enable detection of persons, vehicles, objects
            enable_plate_recognition=True,
            confidence_threshold=0.4  # Lower threshold to catch more objects
        )
        print("✅ Object detector initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        detector = None
        return False

class CameraStream:
    def __init__(self, camera_config):
        self.camera = camera_config
        self.frame = None
        self.frame_with_detections = None  # Frame with detection boxes drawn
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
        
        # Detection stats
        self.last_detections = []
        self.detection_counts = {'Person': 0, 'Vehicle': 0, 'Object': 0}
        
    def get_rtsp_url(self):
        """Generate RTSP URL for the camera"""
        ip = self.camera['ip']
        if CAMERA_PASSWORD:
            return f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{ip}:554/stream1"
        else:
            return f"rtsp://{CAMERA_USERNAME}@{ip}:554/stream1"
    
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
        global detector
        
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
                        frame_with_boxes = self._process_detections(frame)
                        
                        with self.lock:
                            self.frame = frame
                            self.frame_with_detections = frame_with_boxes
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
    
    def _process_detections(self, frame):
        """Run object detection on frame and draw bounding boxes"""
        global detector
        
        if detector is None:
            return frame
        
        try:
            with detector_lock:
                detections = detector.detect(frame, recognize_plates=True)
            
            # Update detection counts
            counts = {'Person': 0, 'Vehicle': 0, 'Object': 0}
            for det in detections:
                if det.label in counts:
                    counts[det.label] += 1
            self.detection_counts = counts
            self.last_detections = detections
            
            # Draw detections on frame
            annotated_frame = detector.draw_detections(
                frame,
                detections,
                show_label=True,
                show_confidence=True,
                show_plate=True
            )
            
            # Add detection count overlay
            y_offset = 25
            for label, count in counts.items():
                if count > 0:
                    text = f"{label}: {count}"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    y_offset += 25
            
            return annotated_frame
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return frame
    
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
        """Get current frame as JPEG - with detections drawn"""
        with self.lock:
            if self.frame_with_detections is None and self.frame is None:
                placeholder = self._create_placeholder()
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            
            # Return frame with detection boxes if available
            frame_to_send = self.frame_with_detections if self.frame_with_detections is not None else self.frame
            if frame_to_send is None or not self.connected:
                placeholder = self._create_placeholder()
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
                
            _, jpeg = cv2.imencode('.jpg', frame_to_send)
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
            'detections': self.detection_counts  # Include detection counts
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
    """Initialize all camera streams"""
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
    return jsonify({
        "status": "healthy", 
        "mode": "demo",
        "cameras_online": camera_status,
        "timestamp": datetime.now().isoformat()
    }), 200

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
    }), 404

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
    print("🔧 Initializing object detector...")
    init_detector()
    
    print("🎥 Initializing cameras...")
    init_cameras()
    
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Starting dashboard on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

