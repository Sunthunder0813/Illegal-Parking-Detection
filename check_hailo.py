#!/usr/bin/env python3
"""
Hailo-8L Diagnostic Script for Raspberry Pi 5
Run this to check if Hailo is properly installed and accessible.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("🔍 HAILO-8L DIAGNOSTIC CHECK")
    print("   For Raspberry Pi 5 with Hailo-8L M.2 HAT (13 TOPS)")
    print("=" * 60)
    
    issues = []
    
    # Check 1: hailortcli
    print("\n1️⃣ Checking hailortcli...")
    try:
        result = subprocess.run(["hailortcli", "fw-control", "identify"], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"   ✅ hailortcli working")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"   ❌ hailortcli failed: {result.stderr}")
            issues.append("hailortcli not working")
    except FileNotFoundError:
        print("   ❌ hailortcli not found")
        issues.append("hailortcli not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        issues.append(f"hailortcli error: {e}")
    
    # Check 2: hailo_platform Python module
    print("\n2️⃣ Checking hailo_platform Python module...")
    try:
        import hailo_platform
        print(f"   ✅ hailo_platform imported")
        print(f"   Location: {hailo_platform.__file__}")
        
        # Try importing specific classes
        from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams
        print("   ✅ All required classes available")
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        issues.append("hailo_platform not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        issues.append(f"hailo_platform error: {e}")
    
    # Check 3: VDevice creation
    print("\n3️⃣ Checking Hailo device access...")
    try:
        from hailo_platform import VDevice
        params = VDevice.create_params()
        target = VDevice(params)
        print("   ✅ VDevice created successfully")
        target.release()
        print("   ✅ Device released")
    except Exception as e:
        print(f"   ❌ Cannot access Hailo device: {e}")
        issues.append(f"Device access failed: {e}")
    
    # Check 4: HEF files
    print("\n4️⃣ Checking for HEF model files...")
    project_dir = Path(__file__).parent
    hef_files = list(project_dir.glob("*.hef"))
    
    if hef_files:
        for hef in hef_files:
            print(f"   ✅ Found: {hef.name} ({hef.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print("   ❌ No HEF files found in project directory")
        issues.append("No HEF file")
        
        # Check system paths
        system_paths = [
            "/usr/share/hailo-models/",
            "/usr/share/hailo/models/",
            Path.home() / "hailo-rpi5-examples/resources/"
        ]
        print("\n   Checking system paths:")
        for p in system_paths:
            p = Path(p)
            if p.exists():
                hefs = list(p.glob("*.hef"))
                if hefs:
                    print(f"   📁 Found in {p}:")
                    for h in hefs[:5]:
                        print(f"      - {h.name}")
    
    # Check 5: Try loading a HEF
    print("\n5️⃣ Testing HEF loading...")
    if hef_files:
        try:
            from hailo_platform import HEF
            hef = HEF(str(hef_files[0]))
            print(f"   ✅ Successfully loaded: {hef_files[0].name}")
        except Exception as e:
            print(f"   ❌ Failed to load HEF: {e}")
            issues.append(f"HEF load failed: {e}")
    else:
        print("   ⏭️ Skipped - no HEF file available")
    
    # Summary
    print("\n" + "=" * 60)
    if not issues:
        print("✅ ALL CHECKS PASSED - Hailo-8L should work!")
        print("\n   Run your dashboard and it should use Hailo acceleration.")
    else:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n📋 TO FIX:")
        if "hailo_platform not installed" in str(issues) or "hailortcli not installed" in str(issues):
            print("   1. Install Hailo software:")
            print("      sudo apt update")
            print("      sudo apt install hailo-all")
            print("      sudo reboot")
        
        if "No HEF file" in str(issues):
            print("\n   2. Get the HEF model file:")
            print("      Option A - From Hailo examples:")
            print("         git clone https://github.com/hailo-ai/hailo-rpi5-examples.git")
            print("         cd hailo-rpi5-examples && ./download_resources.sh")
            print("         cp resources/yolov8n_h8l.hef ~/Illegal-Parking-Detection/")
            print("")
            print("      Option B - Direct download:")
            print("         wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov8n_h8l.hef")
        
        if "Device access failed" in str(issues):
            print("\n   3. Check hardware connection:")
            print("      - Ensure Hailo-8L M.2 HAT is properly seated")
            print("      - Check PCIe connection")
            print("      - Try: sudo reboot")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
