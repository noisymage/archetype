#!/usr/bin/env python3
"""
Diagnostic script to investigate face matching issues.

Tests why an identical reference image shows 69.1% match instead of ~100%.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from vision_engine import ModelManager, FaceAnalyzer
from batch_processor import compute_similarity, compute_pose_distance, compute_pose_aware_similarity


def test_identical_image_matching(image_path: str):
    """
    Test face matching with an identical image.
    
    This simulates adding a reference image to the dataset and processing it.
    """
    print("="*80)
    print("DIAGNOSTIC: Face Matching Logic Test")
    print("="*80)
    print(f"\nTest Image: {image_path}\n")
    
    if not Path(image_path).exists():
        print(f"❌ ERROR: Image not found at {image_path}")
        return
    
    # Initialize models
    print("📦 Loading models...")
    manager = ModelManager()
    analyzer = FaceAnalyzer(manager)
    
    # Analyze the image twice (simulating reference vs. dataset)
    print("🔍 Analyzing image as 'reference'...")
    result_ref = analyzer.analyze(image_path)
    
    print("🔍 Analyzing same image as 'dataset'...")
    result_dataset = analyzer.analyze(image_path)
    
    if not result_ref.detected or not result_dataset.detected:
        print(f"❌ FAILED: Face not detected")
        print(f"   Reference: {result_ref.error}")
        print(f"   Dataset: {result_dataset.error}")
        return
    
    print("✅ Face detected in both analyses\n")
    
    # Extract data
    embedding_ref = result_ref.embedding
    embedding_dataset = result_dataset.embedding
    pose_ref = result_ref.pose
    pose_dataset = result_dataset.pose
    
    # Test 1: Direct cosine similarity
    print("-" * 80)
    print("TEST 1: Direct Embedding Similarity (cosine)")
    print("-" * 80)
    
    direct_similarity = compute_similarity(embedding_ref, embedding_dataset)
    print(f"Similarity: {direct_similarity * 100:.2f}%")
    
    if direct_similarity >= 0.99:
        print("✅ PASS: Near-perfect match (as expected for identical image)")
    elif direct_similarity >= 0.95:
        print("⚠️  WARN: Very high match but not perfect (acceptable)")
    else:
        print(f"❌ FAIL: Unexpectedly low for identical image")
        print("   → Hypothesis: InsightFace may have non-deterministic elements")
    
    # Test 2: Pose distance
    if pose_ref and pose_dataset:
        print("\n" + "-" * 80)
        print("TEST 2: Pose Distance")
        print("-" * 80)
        print(f"Reference pose: yaw={pose_ref['yaw']:.1f}°, pitch={pose_ref['pitch']:.1f}°, roll={pose_ref['roll']:.1f}°")
        print(f"Dataset pose:   yaw={pose_dataset['yaw']:.1f}°, pitch={pose_dataset['pitch']:.1f}°, roll={pose_dataset['roll']:.1f}°")
        
        pose_dist = compute_pose_distance(pose_ref, pose_dataset)
        print(f"\nPose distance: {pose_dist:.2f}°")
        
        if pose_dist < 1.0:
            print("✅ PASS: Near-zero pose difference (as expected)")
        elif pose_dist < 5.0:
            print("⚠️  WARN: Small pose difference detected")
        else:
            print(f"❌ FAIL: Significant pose difference for identical image")
            print("   → Hypothesis: Head pose estimation has variance")
    else:
        print("\n" + "-" * 80)
        print("TEST 2: Pose Distance")
        print("-" * 80)
        print("⚠️  SKIP: Pose data not available")
    
    # Test 3: Master embedding (averaged)
    print("\n" + "-" * 80)
    print("TEST 3: Master Embedding Effect")
    print("-" * 80)
    print("Simulating how reference embeddings are averaged...")
    
    # Simulate averaging with itself (should still be identical)
    master_embedding = np.mean([embedding_ref, embedding_ref], axis=0)
    master_similarity = compute_similarity(master_embedding, embedding_dataset)
    print(f"Master → Dataset similarity: {master_similarity * 100:.2f}%")
    
    if master_similarity >= 0.99:
        print("✅ PASS: Averaging doesn't degrade matching")
    else:
        print("⚠️  INFO: Even self-averaging shows some variance")
    
    # Test 4: Pose-aware similarity
    if pose_ref and pose_dataset:
        print("\n" + "-" * 80)
        print("TEST 4: Pose-Aware Similarity (Current System)")
        print("-" * 80)
        print("This simulates the actual matching used in batch processing...")
        
        reference_data = [(embedding_ref, pose_ref, 1)]  # ref_id=1
        pose_aware_sim, best_ref_id = compute_pose_aware_similarity(
            embedding_dataset,
            pose_dataset,
            reference_data,
            pose_threshold=30.0
        )
        
        print(f"Pose-aware similarity: {pose_aware_sim * 100:.2f}%")
        print(f"Best matching reference: {best_ref_id}")
        
        if pose_aware_sim >= 0.95:
            print("✅ PASS: Pose-aware matching works correctly")
        else:
            print(f"❌ FAIL: Pose-aware matching degrades score")
            print("   → Hypothesis: Weighting formula may be too aggressive")
            
            # Calculate the weight that was applied
            sigma = 15.0
            import math
            weight = math.exp(-pose_dist / sigma) if pose_dist else 1.0
            print(f"   → Pose distance: {pose_dist:.2f}°")
            print(f"   → Applied weight: {weight:.4f}")
            print(f"   → Direct similarity: {direct_similarity * 100:.2f}%")
            print(f"   → Weighted result: {pose_aware_sim * 100:.2f}%")
    
    # Test 5: Multi-reference scenario
    print("\n" + "-" * 80)
    print("TEST 5: Multi-Reference Scenario (Real-World)")
    print("-" * 80)
    print("Simulating multiple reference views with different poses...")
    
    if pose_ref and pose_dataset:
        # Create synthetic references with pose variations
        ref_poses = [
            {"yaw": pose_ref["yaw"], "pitch": pose_ref["pitch"], "roll": pose_ref["roll"]},  # Identical
            {"yaw": pose_ref["yaw"] + 45, "pitch": pose_ref["pitch"], "roll": pose_ref["roll"]},  # 45° different
            {"yaw": pose_ref["yaw"] - 45, "pitch": pose_ref["pitch"], "roll": pose_ref["roll"]},  # 45° opposite
        ]
        
        reference_data_multi = [
            (embedding_ref, ref_poses[0], 1),
            (embedding_ref * 0.98, ref_poses[1], 2),  # Slightly different embedding  
            (embedding_ref * 0.97, ref_poses[2], 3),  # More different
        ]
        
        multi_sim, multi_best = compute_pose_aware_similarity(
            embedding_dataset,
            pose_dataset,
            reference_data_multi,
            pose_threshold=30.0
        )
        
        print(f"Multi-reference similarity: {multi_sim * 100:.2f}%")
        print(f"Best match: Reference #{multi_best}")
        
        if multi_best == 1:
            print("✅ PASS: Correctly identified exact match")
        else:
            print(f"⚠️  WARN: Best match is not the identical reference")
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Direct Embedding Match: {direct_similarity * 100:.2f}%")
    if pose_ref and pose_dataset:
        print(f"📊 Pose Distance: {pose_dist:.2f}°")
        print(f"📊 Pose-Aware Match: {pose_aware_sim * 100:.2f}%")
    
    print("\n🔍 Likely Causes for 69.1% Score:")
    
    if direct_similarity < 0.95:
        print("  1. ❗ InsightFace embeddings have variance across runs")
        print("     → Solution: Average multiple extractions or accept variance")
    
    if pose_ref and pose_dataset and pose_dist > 2.0:
        print("  2. ❗ Head pose estimation variance")
        print("     → Solution: Increase pose threshold or reduce weighting")
    
    if pose_ref and pose_dataset and pose_aware_sim < direct_similarity * 0.95:
        print("  3. ❗ Pose-aware weighting is too aggressive")
        print("     → Solution: Adjust sigma parameter or weighting formula")
    
    # Check if master embedding averaging is the issue
    embeddings_diff = abs(direct_similarity - master_similarity)
    if embeddings_diff > 0.05:
        print("  4. ❗ Master embedding averaging degrades matching")
        print("     → Solution: Use individual reference comparisons instead of average")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test face matching logic")
    parser.add_argument(
        "--ref-image",
        type=str,
        required=True,
        help="Path to reference image to test (e.g., Hanna_27f33096.png)"
    )
    
    args = parser.parse_args()
    test_identical_image_matching(args.ref_image)
