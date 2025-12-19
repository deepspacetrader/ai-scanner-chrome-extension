#!/usr/bin/env python3
import requests
import base64
import json
import os
from pathlib import Path

def test_florence():
    base_url = "http://localhost:8001"
    
    # Find an image in the images directory
    images_dir = Path("images")
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No images found in images/ directory for testing.")
        return False
    
    test_image_path = image_files[0]
    print(f"🧪 Testing Florence-2 with {test_image_path}...")
    
    try:
        with open(test_image_path, "rb") as f:
            image_data = f.read()
        
        base64_image = base64.b64encode(image_data).decode()
        
        # Test 1: Analyze a box (full image for simplicity)
        payload = {
            "image": base64_image,
            "box": {"x": 0, "y": 0, "width": 100, "height": 100},
            "type": "car"
        }
        
        print("📡 Sending request to /api/analyze-box...")
        response = requests.post(f"{base_url}/api/analyze-box", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Florence Analysis Result: {data.get('analysis')}")
            return True
        else:
            print(f"❌ Florence test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Florence test failed: {e}")
        return False

if __name__ == "__main__":
    test_florence()
