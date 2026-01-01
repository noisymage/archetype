#!/usr/bin/env python3
"""Test head pose estimation on a single image."""
import numpy as np
import cv2
from vision_engine import ModelManager, FaceAnalyzer

# Test with a reference image
test_image = "/Users/johan/Development/archetype/backend/test_data/head_front.png"

print(f"Testing pose estimation on: {test_image}")
print("-" * 60)

# Initialize
manager = ModelManager()
analyzer = FaceAnalyzer(manager)

# Analyze the face
result = analyzer.analyze(test_image)

if not result.detected:
    print(f"❌ Face not detected: {result.error}")
    exit(1)

print(f"✅ Face detected!")
print(f"   Confidence: {result.confidence:.3f}")
print(f"   Bbox: {result.bbox}")
print(f"   Landmarks shape: {result.landmarks.shape if result.landmarks is not None else 'None'}")

if result.landmarks is not None:
    print(f"\nFirst 10 landmarks:")
    for i, lm in enumerate(result.landmarks[:10]):
        print(f"   {i}: {lm}")

if result.pose:
    print(f"\n✅ Pose estimated successfully!")
    print(f"   Yaw:   {result.pose['yaw']:.2f}°")
    print(f"   Pitch: {result.pose['pitch']:.2f}°")
    print(f"   Roll:  {result.pose['roll']:.2f}°")
else:
    print(f"\n❌ Pose estimation failed")
    print(f"   Will try to debug...")
    
    # Try to manually estimate pose to see the error
    if result.landmarks is not None:
        from vision_engine import estimate_head_pose
        img = cv2.imread(test_image)
        pose = estimate_head_pose(result.landmarks, img.shape)
        print(f"   Manual attempt result: {pose}")

print("\n" + "=" * 60)
