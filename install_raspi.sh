#!/bin/bash
# Raspberry Pi 5 Installation Script for Illegal Parking Detection

echo "🚀 Installing dependencies for Illegal Parking Detection..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopencv-dev \
    tesseract-ocr

# Create virtual environment
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install flask numpy opencv-python-headless ultralytics easyocr

# Download YOLOv8 model
echo "📥 Downloading YOLOv8 model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Check for Hailo (optional)
if command -v hailortcli &> /dev/null; then
    echo "✅ Hailo runtime detected"
else
    echo "⚠️ Hailo not installed. For GPU acceleration:"
    echo "   sudo apt install hailo-all && sudo reboot"
fi

echo ""
echo "✅ Installation complete!"
echo "📌 To run the dashboard:"
echo "   source venv/bin/activate"
echo "   python dashboard.py"
