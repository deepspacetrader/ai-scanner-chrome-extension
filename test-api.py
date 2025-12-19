#!/usr/bin/env python3
"""
Test script for Ultralytics YOLO Detection API
"""

import requests
import base64
import json
from PIL import Image
import io

def test_api():
    """Test the YOLO API endpoints"""

    base_url = "http://localhost:8001"

    print("🧪 Testing Ultralytics YOLO API...")

    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

    # Test 2: Status check
    try:
        response = requests.get(f"{base_url}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status check passed: {data.get('status')}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False

    # Test 3: Test with a sample image
    try:
        # Load a sample image
        image_path = "images/test.jpg"
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Convert to base64
        base64_image = base64.b64encode(image_data).decode()

        # Make detection request
        request_data = {"image": base64_image}
        response = requests.post(
            f"{base_url}/api/detect-base64",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            objects_found = data.get("total_objects", 0)
            print(f"✅ Detection test passed: {objects_found} objects found")

            # Show detected objects
            if data.get("data"):
                for obj in data["data"][:3]:  # Show first 3 objects
                    print(f"   - {obj.get('type')} at ({obj.get('x')}, {obj.get('y')}) with {obj.get('confidence')*100}% confidence")

        else:
            print(f"❌ Detection test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

    print("🎉 All tests passed! The API is working correctly.")
    return True

if __name__ == "__main__":
    test_api()
