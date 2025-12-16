#!/usr/bin/env python3
"""
GitHub Push Module for Illegal Parking Detection
Automatically pushes detection results to GitHub repository
"""

import subprocess
import os
import json
from datetime import datetime

# =============================================================================
# GITHUB CONFIGURATION
# =============================================================================
GITHUB_REPO = "https://github.com/Sunthunder0813/Illegal-Parking-Detection.git"
GITHUB_EMAIL = "santanderjoseph13@gmail.com"
GITHUB_USERNAME = "Sunthunder0813"

# Directory where detection results will be saved
RESULTS_DIR = "detection_results"
IMAGES_DIR = "detection_images"

def setup_git():
    """Configure git with user credentials"""
    try:
        subprocess.run(["git", "config", "--global", "user.email", GITHUB_EMAIL], check=True)
        subprocess.run(["git", "config", "--global", "user.name", GITHUB_USERNAME], check=True)
        print(f"✅ Git configured for {GITHUB_USERNAME}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to configure git: {e}")
        return False
    except FileNotFoundError:
        print("❌ Git is not installed. Install with: sudo apt install git")
        return False

def init_repo(project_dir):
    """Initialize git repo if not already initialized"""
    git_dir = os.path.join(project_dir, ".git")
    
    if not os.path.exists(git_dir):
        try:
            subprocess.run(["git", "init"], cwd=project_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", GITHUB_REPO], cwd=project_dir, check=True)
            print(f"✅ Git repository initialized")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Git init warning: {e}")
    
    # Create results directories if they don't exist
    results_path = os.path.join(project_dir, RESULTS_DIR)
    images_path = os.path.join(project_dir, IMAGES_DIR)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    return results_path, images_path

def save_detection_result(project_dir, detections, camera_name, frame=None):
    """
    Save detection results to JSON file and optionally save the frame image
    
    Args:
        project_dir: Base project directory
        detections: List of detection dictionaries
        camera_name: Name of the camera
        frame: Optional OpenCV frame to save as image
    
    Returns:
        Tuple of (json_path, image_path or None)
    """
    import cv2
    
    results_path, images_path = init_repo(project_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare detection data
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "camera": camera_name,
        "detection_count": len(detections),
        "detections": []
    }
    
    for det in detections:
        result_data["detections"].append({
            "class_id": det["class_id"],
            "confidence": round(det["conf"], 4),
            "bbox": {
                "x1": det["bbox"][0],
                "y1": det["bbox"][1],
                "x2": det["bbox"][2],
                "y2": det["bbox"][3]
            }
        })
    
    # Save JSON result
    json_filename = f"detection_{camera_name.replace(' ', '_')}_{timestamp}.json"
    json_path = os.path.join(results_path, json_filename)
    
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=2)
    
    # Save image if provided
    image_path = None
    if frame is not None:
        image_filename = f"detection_{camera_name.replace(' ', '_')}_{timestamp}.jpg"
        image_path = os.path.join(images_path, image_filename)
        cv2.imwrite(image_path, frame)
    
    return json_path, image_path

def push_to_github(project_dir, commit_message=None):
    """
    Commit and push all changes to GitHub
    
    Args:
        project_dir: Base project directory
        commit_message: Optional custom commit message
    
    Returns:
        True if successful, False otherwise
    """
    if commit_message is None:
        commit_message = f"Detection results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    try:
        # Add all changes
        subprocess.run(["git", "add", "-A"], cwd=project_dir, check=True)
        
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_dir,
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            print("ℹ️ No changes to commit")
            return True
        
        # Commit changes
        subprocess.run(["git", "commit", "-m", commit_message], cwd=project_dir, check=True)
        
        # Push to GitHub
        # NOTE: You need to set up authentication (SSH key or credential helper)
        subprocess.run(["git", "push", "-u", "origin", "main"], cwd=project_dir, check=True)
        
        print(f"✅ Pushed to GitHub: {commit_message}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git push failed: {e}")
        print("   TIP: Set up SSH key or use: git config --global credential.helper store")
        return False

def push_detection_event(project_dir, detections, camera_name, frame=None, push_immediately=True):
    """
    Convenience function to save and optionally push a detection event
    
    Args:
        project_dir: Base project directory
        detections: List of detection dictionaries
        camera_name: Name of the camera
        frame: Optional OpenCV frame to save
        push_immediately: If True, push to GitHub immediately
    """
    if len(detections) == 0:
        return  # Don't save empty detections
    
    json_path, image_path = save_detection_result(project_dir, detections, camera_name, frame)
    print(f"💾 Saved detection: {os.path.basename(json_path)}")
    
    if push_immediately:
        vehicle_types = [det.get("class_id", "unknown") for det in detections]
        commit_msg = f"Detected {len(detections)} vehicle(s) on {camera_name}"
        push_to_github(project_dir, commit_msg)


# =============================================================================
# SETUP INSTRUCTIONS
# =============================================================================
# 
# Before using this module, set up GitHub authentication on your Raspberry Pi:
#
# OPTION 1: SSH Key (Recommended)
# -------------------------------
# 1. Generate SSH key:
#    ssh-keygen -t ed25519 -C "santanderjoseph13@gmail.com"
#
# 2. Add key to ssh-agent:
#    eval "$(ssh-agent -s)"
#    ssh-add ~/.ssh/id_ed25519
#
# 3. Copy public key:
#    cat ~/.ssh/id_ed25519.pub
#
# 4. Add to GitHub:
#    - Go to GitHub.com -> Settings -> SSH and GPG keys -> New SSH key
#    - Paste your public key
#
# 5. Change remote to SSH:
#    git remote set-url origin git@github.com:Sunthunder0813/Illegal-Parking-Detection.git
#
# OPTION 2: Personal Access Token
# --------------------------------
# 1. Go to GitHub.com -> Settings -> Developer settings -> Personal access tokens
# 2. Generate new token with 'repo' scope
# 3. Use token as password when pushing:
#    git config --global credential.helper store
#    git push (enter username and token as password)
#
# =============================================================================

if __name__ == "__main__":
    # Test the module
    setup_git()
    print("\n📋 GitHub push module loaded successfully")
    print(f"   Repository: {GITHUB_REPO}")
    print(f"   User: {GITHUB_USERNAME}")
