#!/usr/bin/env python3
"""
Raspberry Pi 5 + Hailo-8L Accelerator - Illegal Parking Detection
==================================================================
This module provides vehicle detection with Philippine license plate recognition
for illegal parking detection systems using Hailo-8L AI accelerator.

Features:
- Car and Truck detection using YOLOv8 on Hailo-8L
- Philippine license plate detection and OCR
- Automatic fallback to CPU inference

Usage:
    from models import HailoDetector, Detection, VEHICLE_CLASSES
    
    detector = HailoDetector()
    detections = detector.detect(frame)
    for det in detections:
        print(f"{det.label}: {det.confidence:.2f} Plate: {det.plate_number}")
"""

import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import subprocess
import re
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

# Model paths - Update these paths for your Raspberry Pi setup
MODEL_DIR = Path(__file__).parent.resolve()
HEF_MODEL_PATH = MODEL_DIR / "yolov8n.hef"
PT_MODEL_PATH = MODEL_DIR / "yolov8n.pt"
ONNX_MODEL_PATH = MODEL_DIR / "yolov8n.onnx"

# Default input size for YOLOv8
DEFAULT_INPUT_SIZE = (640, 640)

# Confidence threshold for detections
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# NMS (Non-Maximum Suppression) threshold
DEFAULT_NMS_THRESHOLD = 0.45

# =============================================================================
# COCO Classes - Vehicle Detection Focus
# =============================================================================

# Full COCO class names (80 classes)
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}

# Vehicle classes for illegal parking detection - CARS AND TRUCKS ONLY
VEHICLE_CLASSES = {
    2: "Car",
    7: "Truck"
}

# Extended vehicle classes (if you want to include more vehicle types)
EXTENDED_VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# =============================================================================
# Philippine License Plate Configuration
# =============================================================================

# Philippine License Plate Formats:
# - Private vehicles (new): AAA 1234 (3 letters + space + 4 digits)
# - Private vehicles (old): ABC 123 (3 letters + space + 3 digits)
# - Government vehicles: SGA 1234 or protocol plates
# - Temporary plates: various formats
# - Motorcycle: AA 12345 (2 letters + space + 5 digits)

# Regex patterns for Philippine license plates
PH_PLATE_PATTERNS = [
    # New format: 3 letters + 4 digits (e.g., ABC 1234, NCR 5678)
    r'^[A-Z]{3}[\s\-]?[0-9]{4}$',
    # Old format: 3 letters + 3 digits (e.g., ABC 123)
    r'^[A-Z]{3}[\s\-]?[0-9]{3}$',
    # Government plates
    r'^[S][A-Z]{2}[\s\-]?[0-9]{3,4}$',
    # Motorcycle: 2 letters + 5 digits
    r'^[A-Z]{2}[\s\-]?[0-9]{5}$',
    # Temporary plates
    r'^[0-9]{4}[\s\-]?[0-9]{6}$',
    # Diplomatic plates
    r'^[0-9]{1,3}[\s\-]?[0-9]{1,4}$',
]

# Characters commonly confused in OCR
OCR_CHAR_CORRECTIONS = {
    '0': 'O', 'O': '0',  # Zero vs O
    '1': 'I', 'I': '1',  # One vs I
    '5': 'S', 'S': '5',  # Five vs S
    '8': 'B', 'B': '8',  # Eight vs B
    '2': 'Z', 'Z': '2',  # Two vs Z
    '6': 'G', 'G': '6',  # Six vs G
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PlateDetection:
    """Represents a detected license plate"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) relative to vehicle crop
    plate_number: str  # Recognized plate text
    confidence: float  # OCR confidence
    plate_image: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict:
        return {
            'bbox': self.bbox,
            'plate_number': self.plate_number,
            'confidence': self.confidence
        }


@dataclass
class Detection:
    """Represents a single vehicle detection with optional plate info"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    label: str
    plate_number: Optional[str] = None  # License plate number if detected
    plate_confidence: float = 0.0  # Plate OCR confidence
    plate_bbox: Optional[Tuple[int, int, int, int]] = None  # Plate bounding box
    timestamp: Optional[datetime] = None  # Detection timestamp
    
    @property
    def x1(self) -> int:
        return self.bbox[0]
    
    @property
    def y1(self) -> int:
        return self.bbox[1]
    
    @property
    def x2(self) -> int:
        return self.bbox[2]
    
    @property
    def y2(self) -> int:
        return self.bbox[3]
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def has_plate(self) -> bool:
        return self.plate_number is not None and len(self.plate_number) > 0
    
    def to_dict(self) -> Dict:
        """Convert detection to dictionary"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'label': self.label,
            'plate_number': self.plate_number,
            'plate_confidence': self.plate_confidence,
            'plate_bbox': self.plate_bbox,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class InferenceResult:
    """Represents the complete inference result for a frame"""
    detections: List[Detection]
    inference_time_ms: float
    frame_shape: Tuple[int, int, int]
    
    @property
    def vehicle_count(self) -> int:
        return sum(1 for d in self.detections if d.class_id in VEHICLE_CLASSES)
    
    @property
    def vehicles(self) -> List[Detection]:
        return [d for d in self.detections if d.class_id in VEHICLE_CLASSES]
    
    @property
    def vehicles_with_plates(self) -> List[Detection]:
        return [d for d in self.detections if d.class_id in VEHICLE_CLASSES and d.has_plate]
    
    @property
    def plate_numbers(self) -> List[str]:
        return [d.plate_number for d in self.detections if d.has_plate]


# =============================================================================
# Hailo Platform Detection
# =============================================================================

def check_hailo_available() -> bool:
    """Check if Hailo platform is available"""
    try:
        from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
        return True
    except ImportError:
        return False


def download_hef_model(target_path: Path) -> bool:
    """
    Attempt to download or locate the HEF model file.
    
    Args:
        target_path: Where to save the HEF file
        
    Returns:
        True if successful, False otherwise
    """
    print("🔍 Attempting to find/download HEF file...")
    
    # Option 1: Check system-installed models
    system_hef_paths = [
        "/usr/share/hailo-models/yolov8n.hef",
        "/usr/share/hailo/models/yolov8n.hef",
        "/opt/hailo/models/yolov8n.hef",
        Path.home() / ".hailo" / "models" / "yolov8n.hef"
    ]
    
    for path in system_hef_paths:
        path = Path(path)
        if path.is_file():
            print(f"✅ Found system HEF: {path}")
            try:
                import shutil
                shutil.copy(path, target_path)
                print(f"✅ Copied to: {target_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to copy: {e}")
    
    # Option 2: Try to download from Hailo Model Zoo
    hef_urls = [
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8l/yolov8n.hef",
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov8n_h8l.hef"
    ]
    
    for url in hef_urls:
        print(f"📥 Trying to download from: {url}")
        try:
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(target_path), url],
                timeout=120,
                capture_output=True
            )
            if result.returncode == 0 and target_path.is_file() and target_path.stat().st_size > 1000:
                print(f"✅ Downloaded HEF successfully!")
                return True
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
    
    return False


# =============================================================================
# Philippine License Plate Recognition
# =============================================================================

class PhilippinePlateRecognizer:
    """
    License plate detection and OCR for Philippine plates.
    
    Supports:
    - New format: AAA 1234 (3 letters + 4 digits)
    - Old format: ABC 123 (3 letters + 3 digits)
    - Motorcycle: AA 12345 (2 letters + 5 digits)
    - Government/Special plates
    
    Usage:
        recognizer = PhilippinePlateRecognizer()
        plate_text, confidence = recognizer.recognize(vehicle_crop)
    """
    
    def __init__(self, use_easyocr: bool = True):
        """
        Initialize the plate recognizer.
        
        Args:
            use_easyocr: Use EasyOCR (recommended) or Tesseract
        """
        self.use_easyocr = use_easyocr
        self._ocr_reader = None
        self._tesseract_available = False
        
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize OCR engine"""
        if self.use_easyocr:
            try:
                import easyocr
                # Use English for alphanumeric plates
                self._ocr_reader = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR initialized for plate recognition")
                return
            except ImportError:
                print("⚠️ EasyOCR not found, trying Tesseract...")
            except Exception as e:
                print(f"⚠️ EasyOCR init error: {e}")
        
        # Fallback to Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            print("✅ Tesseract OCR available for plate recognition")
        except Exception as e:
            print(f"⚠️ Tesseract not available: {e}")
            print("   Install with: sudo apt install tesseract-ocr")
            print("   Or: pip install easyocr")
    
    def preprocess_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR accuracy.
        
        Args:
            plate_img: Cropped plate image (BGR)
            
        Returns:
            Preprocessed grayscale image
        """
        if plate_img is None or plate_img.size == 0:
            return None
        
        # Resize for better OCR (standard plate aspect ratio ~4.5:1)
        h, w = plate_img.shape[:2]
        if w < 100:
            scale = 200 / w
            plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        # Increase contrast
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
        
        # Apply bilateral filter to reduce noise while keeping edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh
    
    def detect_plate_region(self, vehicle_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect license plate region within a vehicle image.
        
        Args:
            vehicle_img: Cropped vehicle image (BGR)
            
        Returns:
            Plate bounding box (x1, y1, x2, y2) or None
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return None
        
        h, w = vehicle_img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edges = cv2.Canny(gray, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        plate_contour = None
        
        for contour in contours:
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # License plates typically have 4 corners
            if len(approx) >= 4 and len(approx) <= 6:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / ch if ch > 0 else 0
                
                # Philippine plates have aspect ratio ~4.5:1 to 2:1
                if 1.5 <= aspect_ratio <= 6.0:
                    # Plate is usually in lower half of vehicle
                    if y > h * 0.3:
                        # Check size constraints
                        area_ratio = (cw * ch) / (w * h)
                        if 0.01 <= area_ratio <= 0.3:
                            plate_contour = (x, y, x + cw, y + ch)
                            break
        
        # If no contour found, try template-based detection
        if plate_contour is None:
            # Look in the lower portion of the vehicle
            search_region = vehicle_img[int(h*0.4):, :]
            
            # Try morphological approach
            gray_search = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
            
            # Blackhat to find dark regions on light background (plate characters)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
            blackhat = cv2.morphologyEx(gray_search, cv2.MORPH_BLACKHAT, kernel)
            
            # Threshold
            _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Find contours in threshold image
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                # Get bounding box of all contours
                all_pts = np.vstack(cnts)
                x, y, cw, ch = cv2.boundingRect(all_pts)
                
                # Adjust for search region offset
                y_offset = int(h * 0.4)
                
                # Add padding
                pad = 10
                x1 = max(0, x - pad)
                y1 = max(0, y + y_offset - pad)
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + y_offset + ch + pad)
                
                if (x2 - x1) > 30 and (y2 - y1) > 10:
                    plate_contour = (x1, y1, x2, y2)
        
        return plate_contour
    
    def validate_plate_format(self, text: str) -> Tuple[bool, str]:
        """
        Validate and clean Philippine plate number format.
        
        Args:
            text: Raw OCR text
            
        Returns:
            (is_valid, cleaned_text)
        """
        # Clean the text
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9\s\-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Try to match against Philippine plate patterns
        for pattern in PH_PLATE_PATTERNS:
            clean_text = text.replace(' ', '').replace('-', '')
            if re.match(pattern.replace(r'[\s\-]?', ''), clean_text):
                # Format nicely
                if len(clean_text) == 7:  # AAA1234
                    formatted = f"{clean_text[:3]} {clean_text[3:]}"
                    return True, formatted
                elif len(clean_text) == 6:  # AAA123
                    formatted = f"{clean_text[:3]} {clean_text[3:]}"
                    return True, formatted
                return True, text
        
        # If no pattern matched but looks like a plate, return it anyway
        if len(text) >= 5 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
            return True, text
        
        return False, text
    
    def recognize(self, plate_img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize text from a plate image.
        
        Args:
            plate_img: Cropped plate image (BGR)
            
        Returns:
            (plate_text, confidence) or (None, 0.0)
        """
        if plate_img is None or plate_img.size == 0:
            return None, 0.0
        
        # Preprocess
        processed = self.preprocess_plate_image(plate_img)
        if processed is None:
            return None, 0.0
        
        try:
            if self._ocr_reader is not None:
                # Use EasyOCR
                results = self._ocr_reader.readtext(
                    processed,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -',
                    paragraph=True,
                    min_size=10
                )
                
                if results:
                    # Combine all detected text
                    texts = [r[1] for r in results]
                    confidences = [r[2] for r in results]
                    
                    combined_text = ' '.join(texts)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    # Validate format
                    is_valid, cleaned = self.validate_plate_format(combined_text)
                    
                    return cleaned if cleaned else None, avg_confidence
                    
            elif self._tesseract_available:
                # Use Tesseract
                import pytesseract
                
                # Configure Tesseract for plate recognition
                config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                
                text = pytesseract.image_to_string(processed, config=config)
                
                # Validate format
                is_valid, cleaned = self.validate_plate_format(text)
                
                # Tesseract doesn't provide confidence easily, estimate from result
                confidence = 0.7 if is_valid else 0.3
                
                return cleaned if cleaned else None, confidence
                
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
        
        return None, 0.0
    
    def recognize_from_vehicle(self, vehicle_img: np.ndarray) -> Tuple[Optional[str], float, Optional[Tuple[int, int, int, int]]]:
        """
        Detect and recognize plate from a vehicle image.
        
        Args:
            vehicle_img: Cropped vehicle image (BGR)
            
        Returns:
            (plate_text, confidence, plate_bbox) or (None, 0.0, None)
        """
        # Detect plate region
        plate_bbox = self.detect_plate_region(vehicle_img)
        
        if plate_bbox is None:
            # Try OCR on lower portion of vehicle
            h = vehicle_img.shape[0]
            lower_half = vehicle_img[int(h*0.5):, :]
            text, conf = self.recognize(lower_half)
            return text, conf, None
        
        # Crop plate region
        x1, y1, x2, y2 = plate_bbox
        plate_crop = vehicle_img[y1:y2, x1:x2]
        
        # Recognize
        text, conf = self.recognize(plate_crop)
        
        return text, conf, plate_bbox


# =============================================================================
# Main Detector Class
# =============================================================================

class HailoDetector:
    """
    Unified object detector that uses Hailo-8L accelerator when available,
    with automatic fallback to CPU inference using Ultralytics YOLO.
    
    Includes integrated Philippine license plate recognition.
    
    Example:
        detector = HailoDetector(enable_plate_recognition=True)
        detections = detector.detect(frame)
        for det in detections:
            print(f"{det.label}: Plate {det.plate_number}")
    """
    
    def __init__(
        self,
        hef_path: Optional[str] = None,
        pt_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        target_classes: Optional[Dict[int, str]] = None,
        auto_download: bool = True,
        enable_plate_recognition: bool = True
    ):
        """
        Initialize the detector.
        
        Args:
            hef_path: Path to the HEF model file (for Hailo)
            pt_path: Path to the PyTorch model file (fallback)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            target_classes: Dictionary of class_id -> class_name to detect (default: Car, Truck)
            auto_download: Attempt to download HEF if missing
            enable_plate_recognition: Enable Philippine license plate OCR
        """
        self.hef_path = Path(hef_path) if hef_path else HEF_MODEL_PATH
        self.pt_path = Path(pt_path) if pt_path else PT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = target_classes or VEHICLE_CLASSES  # Default: Car and Truck only
        self.enable_plate_recognition = enable_plate_recognition
        
        # Runtime state
        self.hailo_available = False
        self.input_height = DEFAULT_INPUT_SIZE[0]
        self.input_width = DEFAULT_INPUT_SIZE[1]
        
        # Hailo objects
        self._hef = None
        self._target = None
        self._network_group = None
        self._network_group_params = None
        
        # CPU fallback
        self._yolo_model = None
        
        # Plate recognizer
        self._plate_recognizer = None
        
        # Initialize
        self._initialize(auto_download)
    
    def _initialize(self, auto_download: bool = True):
        """Initialize the appropriate inference backend"""
        
        # Try Hailo first
        if check_hailo_available():
            if not self.hef_path.is_file() and auto_download:
                print("⚠️ HEF file not found. Attempting auto-download...")
                download_hef_model(self.hef_path)
            
            if self.hef_path.is_file():
                try:
                    self._init_hailo()
                    self.hailo_available = True
                    print(f"✅ Hailo-8L initialized successfully")
                    print(f"   Input shape: ({self.input_height}, {self.input_width})")
                    return
                except Exception as e:
                    print(f"❌ Failed to initialize Hailo: {e}")
        else:
            print("⚠️ Hailo platform not available")
            print("   Install with: sudo apt install hailo-all && sudo reboot")
        
        # Fallback to CPU
        self._init_cpu_fallback()
        
        # Initialize plate recognition
        if self.enable_plate_recognition:
            self._init_plate_recognition()
    
    def _init_plate_recognition(self):
        """Initialize Philippine license plate recognizer"""
        try:
            self._plate_recognizer = PhilippinePlateRecognizer()
            print("✅ Philippine plate recognition enabled")
        except Exception as e:
            print(f"⚠️ Plate recognition init failed: {e}")
            self._plate_recognizer = None
    
    def _init_hailo(self):
        """Initialize Hailo inference pipeline"""
        from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams
        
        self._hef = HEF(str(self.hef_path))
        params = VDevice.create_params()
        self._target = VDevice(params)
        
        configure_params = ConfigureParams.create_from_hef(
            self._hef, 
            interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._target.configure(self._hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()
        
        # Get input dimensions
        input_info = self._network_group.get_input_vstream_infos()[0]
        self.input_height = input_info.shape[1]
        self.input_width = input_info.shape[2]
        self._input_name = input_info.name
        
        # Initialize plate recognition for Hailo too
        if self.enable_plate_recognition:
            self._init_plate_recognition()
    
    def _init_cpu_fallback(self):
        """Initialize CPU-based YOLO inference"""
        print("⚠️ Setting up CPU inference (Ultralytics YOLO)")
        print("   WARNING: CPU inference on Pi 5 is ~1-2 FPS. Get a HEF file for ~30+ FPS!")
        
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(self.pt_path))
            print(f"✅ Loaded YOLO model: {self.pt_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for Hailo inference.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Resize to network input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension (HWC -> NHWC)
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def postprocess(
        self, 
        raw_output: np.ndarray, 
        original_shape: Tuple[int, int]
    ) -> List[Detection]:
        """
        Post-process Hailo inference output to get detections.
        
        Args:
            raw_output: Raw output tensor from Hailo
            original_shape: (height, width) of the original frame
            
        Returns:
            List of Detection objects
        """
        detections = []
        orig_h, orig_w = original_shape
        
        # YOLOv8 output format varies based on HEF compilation
        # Common format: [batch, num_detections, 6] where 6 = [x, y, w, h, conf, class]
        # Or: [batch, 84, 8400] which needs transposing
        
        if len(raw_output.shape) == 3:
            # Handle [batch, features, anchors] format
            if raw_output.shape[1] == 84:  # 4 box coords + 80 classes
                raw_output = raw_output[0].T  # Transpose to [anchors, features]
        elif len(raw_output.shape) == 2:
            pass  # Already in correct shape
        else:
            # Flatten and try to parse
            raw_output = raw_output.reshape(-1, raw_output.shape[-1])
        
        for detection in raw_output:
            if len(detection) >= 6:
                # Standard format: x_center, y_center, width, height, confidence, class_id
                x_center, y_center, width, height = detection[:4]
                conf = detection[4]
                cls_id = int(detection[5])
                
                # Apply confidence threshold
                if conf < self.confidence_threshold:
                    continue
                
                # Filter by target classes
                if cls_id not in self.target_classes:
                    continue
                
                # Convert to pixel coordinates
                x1 = int((x_center - width / 2) * orig_w)
                y1 = int((y_center - height / 2) * orig_h)
                x2 = int((x_center + width / 2) * orig_w)
                y2 = int((y_center + height / 2) * orig_h)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))
                
                label = self.target_classes.get(cls_id, COCO_CLASSES.get(cls_id, "Unknown"))
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=cls_id,
                    label=label
                ))
            
            elif len(detection) >= 85:
                # YOLOv8 format: [x, y, w, h, class_scores...]
                x_center, y_center, width, height = detection[:4]
                class_scores = detection[4:84]
                cls_id = int(np.argmax(class_scores))
                conf = float(class_scores[cls_id])
                
                if conf < self.confidence_threshold:
                    continue
                
                if cls_id not in self.target_classes:
                    continue
                
                x1 = int((x_center - width / 2) * orig_w)
                y1 = int((y_center - height / 2) * orig_h)
                x2 = int((x_center + width / 2) * orig_w)
                y2 = int((y_center + height / 2) * orig_h)
                
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))
                
                label = self.target_classes.get(cls_id, COCO_CLASSES.get(cls_id, "Unknown"))
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    label=label
                ))
        
        # Apply NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def _run_hailo_inference(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on Hailo accelerator"""
        from hailo_platform import InferVStreams
        
        input_data = self.preprocess(frame)
        
        with InferVStreams(self._network_group, self._network_group_params) as infer_pipeline:
            input_dict = {self._input_name: input_data}
            output = infer_pipeline.infer(input_dict)
        
        # Get raw output
        output_name = list(output.keys())[0]
        raw_output = output[output_name]
        
        # Post-process
        return self.postprocess(raw_output, frame.shape[:2])
    
    def _run_cpu_inference(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on CPU using Ultralytics YOLO"""
        results = self._yolo_model(frame, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < self.confidence_threshold:
                    continue
                
                if cls_id not in self.target_classes:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.target_classes.get(cls_id, COCO_CLASSES.get(cls_id, "Unknown"))
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    label=label
                ))
        
        return detections
    
    def detect(self, frame: np.ndarray, recognize_plates: bool = True) -> List[Detection]:
        """
        Run object detection on a frame with optional plate recognition.
        
        Args:
            frame: BGR image from OpenCV
            recognize_plates: Whether to run plate OCR on detected vehicles
            
        Returns:
            List of Detection objects with plate info if available
        """
        try:
            if self.hailo_available:
                detections = self._run_hailo_inference(frame)
            else:
                detections = self._run_cpu_inference(frame)
            
            # Run plate recognition on each detection
            if recognize_plates and self._plate_recognizer is not None:
                detections = self._recognize_plates(frame, detections)
            
            # Add timestamp to all detections
            now = datetime.now()
            for det in detections:
                det.timestamp = now
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            return []
    
    def _recognize_plates(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        """
        Run plate recognition on detected vehicles.
        
        Args:
            frame: Original frame
            detections: List of vehicle detections
            
        Returns:
            Detections with plate info populated
        """
        for det in detections:
            try:
                # Crop vehicle region
                x1, y1, x2, y2 = det.bbox
                vehicle_crop = frame[y1:y2, x1:x2]
                
                if vehicle_crop.size == 0:
                    continue
                
                # Recognize plate
                plate_text, confidence, plate_bbox = self._plate_recognizer.recognize_from_vehicle(vehicle_crop)
                
                if plate_text:
                    det.plate_number = plate_text
                    det.plate_confidence = confidence
                    
                    # Convert plate bbox to absolute coordinates
                    if plate_bbox:
                        px1, py1, px2, py2 = plate_bbox
                        det.plate_bbox = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
                        
            except Exception as e:
                # Don't fail detection if plate recognition fails
                pass
        
        return detections
    
    def detect_with_timing(self, frame: np.ndarray, recognize_plates: bool = True) -> InferenceResult:
        """
        Run detection with timing information.
        
        Args:
            frame: BGR image from OpenCV
            recognize_plates: Whether to run plate OCR
            
        Returns:
            InferenceResult with detections and timing
        """
        import time
        start = time.perf_counter()
        detections = self.detect(frame, recognize_plates=recognize_plates)
        elapsed = (time.perf_counter() - start) * 1000
        
        return InferenceResult(
            detections=detections,
            inference_time_ms=elapsed,
            frame_shape=frame.shape
        )
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
        plate_color: Tuple[int, int, int] = (0, 165, 255),  # Orange for plates
        thickness: int = 2,
        show_label: bool = True,
        show_confidence: bool = True,
        show_plate: bool = True
    ) -> np.ndarray:
        """
        Draw detection boxes and plate numbers on frame.
        
        Args:
            frame: BGR image
            detections: List of Detection objects
            color: Vehicle box color (BGR)
            plate_color: Plate box/text color (BGR)
            thickness: Line thickness
            show_label: Show class label
            show_confidence: Show confidence score
            show_plate: Show plate number
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw vehicle box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Build label text (vehicle type + confidence)
            if show_label or show_confidence:
                parts = []
                if show_label:
                    parts.append(det.label)
                if show_confidence:
                    parts.append(f"{det.confidence:.2f}")
                label_text = " ".join(parts)
                
                # Calculate label position
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width + 4, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated,
                    label_text,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # Draw plate info
            if show_plate and det.has_plate:
                # Draw plate bounding box if available
                if det.plate_bbox:
                    px1, py1, px2, py2 = det.plate_bbox
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), plate_color, 2)
                
                # Draw plate number below vehicle box
                plate_text = f"🚗 {det.plate_number}"
                (plate_w, plate_h), _ = cv2.getTextSize(
                    det.plate_number, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                # Background for plate text
                cv2.rectangle(
                    annotated,
                    (x1, y2 + 2),
                    (x1 + plate_w + 10, y2 + plate_h + 12),
                    plate_color,
                    -1
                )
                
                # Plate text
                cv2.putText(
                    annotated,
                    det.plate_number,
                    (x1 + 5, y2 + plate_h + 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        
        return annotated
    
    def release(self):
        """Release Hailo resources"""
        if self._target is not None:
            try:
                self._target.release()
            except Exception as e:
                print(f"⚠️ Error releasing Hailo device: {e}")
            self._target = None
    
    def __del__(self):
        """Destructor - release resources"""
        self.release()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
        return False


# =============================================================================
# Parking Zone Detection
# =============================================================================

class ParkingZone:
    """Represents a parking zone for illegal parking detection"""
    
    def __init__(
        self,
        zone_id: str,
        polygon: List[Tuple[int, int]],
        name: str = "",
        is_no_parking: bool = True
    ):
        """
        Initialize a parking zone.
        
        Args:
            zone_id: Unique identifier for the zone
            polygon: List of (x, y) points defining the zone boundary
            name: Human-readable name for the zone
            is_no_parking: True if parking is prohibited in this zone
        """
        self.zone_id = zone_id
        self.polygon = np.array(polygon, dtype=np.int32)
        self.name = name or zone_id
        self.is_no_parking = is_no_parking
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the zone"""
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0
    
    def contains_detection(self, detection: Detection) -> bool:
        """Check if the center of a detection is inside the zone"""
        return self.contains_point(detection.center)
    
    def get_overlap_ratio(self, detection: Detection) -> float:
        """Calculate how much of a detection overlaps with the zone"""
        x1, y1, x2, y2 = detection.bbox
        
        # Create detection rectangle mask
        det_mask = np.zeros((max(y2, self.polygon[:, 1].max()) + 1,
                            max(x2, self.polygon[:, 0].max()) + 1), dtype=np.uint8)
        cv2.rectangle(det_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Create zone mask
        zone_mask = np.zeros_like(det_mask)
        cv2.fillPoly(zone_mask, [self.polygon], 255)
        
        # Calculate intersection
        intersection = cv2.bitwise_and(det_mask, zone_mask)
        
        det_area = detection.area
        if det_area == 0:
            return 0.0
        
        intersection_area = np.count_nonzero(intersection)
        return intersection_area / det_area
    
    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = None,
        alpha: float = 0.3,
        show_name: bool = True
    ) -> np.ndarray:
        """Draw the zone on a frame"""
        if color is None:
            color = (0, 0, 255) if self.is_no_parking else (0, 255, 0)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw border
        cv2.polylines(frame, [self.polygon], True, color, 2)
        
        # Draw name
        if show_name:
            centroid = self.polygon.mean(axis=0).astype(int)
            cv2.putText(
                frame,
                self.name,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return frame


class IllegalParkingDetector:
    """
    Combines object detection with parking zone analysis
    to detect illegal parking violations.
    """
    
    def __init__(
        self,
        detector: HailoDetector,
        zones: List[ParkingZone] = None,
        overlap_threshold: float = 0.5,
        dwell_time_seconds: float = 5.0
    ):
        """
        Initialize illegal parking detector.
        
        Args:
            detector: HailoDetector instance for vehicle detection
            zones: List of ParkingZone objects
            overlap_threshold: Minimum overlap ratio to consider a violation
            dwell_time_seconds: How long a vehicle must be stationary to be a violation
        """
        self.detector = detector
        self.zones = zones or []
        self.overlap_threshold = overlap_threshold
        self.dwell_time_seconds = dwell_time_seconds
        
        # Track vehicles over time for dwell detection
        self._tracked_vehicles: Dict[str, Dict] = {}
    
    def add_zone(self, zone: ParkingZone):
        """Add a parking zone"""
        self.zones.append(zone)
    
    def detect_violations(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect parking violations in a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            List of violation dictionaries with detection and zone info
        """
        detections = self.detector.detect(frame)
        violations = []
        
        for det in detections:
            for zone in self.zones:
                if zone.is_no_parking:
                    overlap = zone.get_overlap_ratio(det)
                    if overlap >= self.overlap_threshold:
                        violations.append({
                            'detection': det,
                            'zone': zone,
                            'overlap_ratio': overlap,
                            'timestamp': None  # Caller should set timestamp
                        })
        
        return violations
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all zones on frame"""
        for zone in self.zones:
            frame = zone.draw(frame)
        return frame


# =============================================================================
# Utility Functions
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    info = {
        'hailo_available': check_hailo_available(),
        'hef_exists': HEF_MODEL_PATH.is_file(),
        'pt_exists': PT_MODEL_PATH.is_file(),
    }
    
    # Check Hailo device
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=10
        )
        info['hailo_device'] = result.stdout.strip() if result.returncode == 0 else None
    except:
        info['hailo_device'] = None
    
    return info


def benchmark_detector(detector: HailoDetector, num_frames: int = 100) -> Dict[str, float]:
    """
    Benchmark detector performance.
    
    Args:
        detector: HailoDetector instance
        num_frames: Number of frames to benchmark
        
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    # Create random test frame
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    times = []
    for _ in range(num_frames):
        start = time.perf_counter()
        detector.detect(test_frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    times = np.array(times)
    
    return {
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'fps': float(1000 / times.mean()) if times.mean() > 0 else 0
    }


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚗 Illegal Parking Detection - Philippine Plate Recognition")
    print("=" * 60)
    
    # Print system info
    print("\n📊 System Information:")
    info = get_system_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Print target classes
    print(f"\n🎯 Target Classes: {VEHICLE_CLASSES}")
    
    # Initialize detector
    print("\n🔧 Initializing detector...")
    try:
        with HailoDetector(enable_plate_recognition=True) as detector:
            print(f"   Using Hailo: {detector.hailo_available}")
            print(f"   Plate Recognition: {detector._plate_recognizer is not None}")
            
            # Create test frame
            print("\n🎬 Testing detection...")
            test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 255, 255), -1)
            
            # Test detection
            result = detector.detect_with_timing(test_frame)
            print(f"   Detections: {len(result.detections)}")
            print(f"   Vehicles with plates: {len(result.vehicles_with_plates)}")
            print(f"   Inference time: {result.inference_time_ms:.2f} ms")
            
            # Print any detected plates
            for det in result.detections:
                print(f"   - {det.label}: Plate={det.plate_number or 'N/A'}")
            
            # Benchmark
            print("\n⏱️ Benchmarking (10 frames, no plate OCR)...")
            stats = benchmark_detector(detector, num_frames=10)
            print(f"   Mean: {stats['mean_ms']:.2f} ms")
            print(f"   FPS: {stats['fps']:.1f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print usage example
    print("\n" + "=" * 60)
    print("📖 Usage Example:")
    print("=" * 60)
    print("""
from models import HailoDetector, VEHICLE_CLASSES

# Initialize detector (auto-detects Hailo or falls back to CPU)
detector = HailoDetector(enable_plate_recognition=True)

# Capture frame from camera
frame = cv2.imread('test_image.jpg')

# Detect vehicles and plates
detections = detector.detect(frame)

for det in detections:
    print(f"Vehicle: {det.label}")
    print(f"  Confidence: {det.confidence:.2f}")
    print(f"  Plate: {det.plate_number or 'Not detected'}")
    print(f"  Location: {det.bbox}")

# Draw results on frame
annotated = detector.draw_detections(frame, detections)
cv2.imshow('Detections', annotated)
""")
    
    print("\n✅ Test complete!")
