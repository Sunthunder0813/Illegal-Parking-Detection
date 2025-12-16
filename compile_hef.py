#!/usr/bin/env python3
"""
HEF Compilation Script for Hailo-8L
Run this on an x86 Ubuntu machine with Hailo Dataflow Compiler installed.

Steps:
1. Install Hailo DFC from Hailo Developer Zone
2. Place your yolov8n.onnx in the same directory
3. Run: python compile_hef.py
4. Transfer the resulting yolov8n.hef to your Raspberry Pi
"""

import subprocess
import os
import sys

# Paths
ONNX_MODEL = "yolov8n.onnx"
HAR_MODEL = "yolov8n.har"
HEF_MODEL = "yolov8n.hef"

# Hailo-8L target (hw_arch for AI Kit on Raspberry Pi 5)
TARGET_HW = "hailo8l"

def check_dependencies():
    """Check if Hailo DFC is installed"""
    try:
        result = subprocess.run(["hailo", "--version"], capture_output=True, text=True)
        print(f"✅ Hailo DFC found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Hailo DFC not found. Please install it from Hailo Developer Zone.")
        return False

def parse_onnx():
    """Parse ONNX model to HAR format"""
    print(f"📦 Parsing {ONNX_MODEL} to HAR...")
    cmd = [
        "hailo", "parser", "onnx",
        ONNX_MODEL,
        "--hw-arch", TARGET_HW,
        "--output-har-path", HAR_MODEL
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Parse failed:\n{result.stderr}")
        sys.exit(1)
    print("✅ HAR file created")

def optimize_model():
    """Optimize the HAR model"""
    print("⚙️ Optimizing model...")
    # Create a simple optimization script
    optimize_script = """
import hailo_sdk_client
from hailo_sdk_client import ClientRunner

runner = ClientRunner(har="yolov8n.har", hw_arch="hailo8l")

# Use default calibration (for better accuracy, provide calibration images)
runner.optimize(calib_set_path=None)

# Save optimized HAR
runner.save_har("yolov8n_optimized.har")
print("Optimization complete")
"""
    with open("optimize_temp.py", "w") as f:
        f.write(optimize_script)
    
    result = subprocess.run(["python", "optimize_temp.py"], capture_output=True, text=True)
    os.remove("optimize_temp.py")
    
    if result.returncode != 0:
        print(f"⚠️ Optimization warning:\n{result.stderr}")
        # Continue anyway, use non-optimized HAR
        return HAR_MODEL
    return "yolov8n_optimized.har"

def compile_hef(har_path):
    """Compile HAR to HEF"""
    print(f"🔧 Compiling {har_path} to HEF...")
    cmd = [
        "hailo", "compiler",
        har_path,
        "--hw-arch", TARGET_HW,
        "--output-dir", ".",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Compilation failed:\n{result.stderr}")
        sys.exit(1)
    print(f"✅ HEF file created: {HEF_MODEL}")

def main():
    if not os.path.exists(ONNX_MODEL):
        print(f"❌ {ONNX_MODEL} not found in current directory")
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    parse_onnx()
    optimized_har = optimize_model()
    compile_hef(optimized_har)
    
    print(f"\n🎉 Done! Transfer {HEF_MODEL} to your Raspberry Pi:")
    print(f"   scp {HEF_MODEL} set-admin@<pi-ip>:/home/set-admin/Illegal-Parking-Detection/")

if __name__ == "__main__":
    main()
