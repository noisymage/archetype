#!/usr/bin/env python3
"""
Test the exact scenario from the database:
- Load all 11 reference embeddings
- Compare dataset image against them using pose-aware matching
- See if we can reproduce the 69.1% score
"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from database import SessionLocal, ReferenceImage
from batch_processor import compute_pose_aware_similarity

db = SessionLocal()

try:
    print("=" * 80)
    print("Reproducing 69.1% Score from Database References")
    print("=" * 80)
    
    # Get all reference images for character ID 1 (Hanna)
    refs = db.query(ReferenceImage).filter(
        ReferenceImage.character_id == 1,
        ReferenceImage.embedding_blob.isnot(None)
    ).all()
    
    print(f"\nFound {len(refs)} reference images with embeddings:\n")
    
    # Build reference data list
    reference_data = []
    ref_6_embedding = None
    ref_6_pose = None
    
    for ref in refs:
        embedding = np.frombuffer(ref.embedding_blob, dtype=np.float32)
        pose = json.loads(ref.pose_json) if ref.pose_json else None
        
        print(f"  Ref #{ref.id} ({ref.view_type}): {Path(ref.path).name}")
        if pose:
            print(f"    Pose: yaw={pose['yaw']:.1f}°, pitch={pose['pitch']:.1f}°, roll={pose['roll']:.1f}°")
        print(f"    Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
        
        if pose:
            reference_data.append((embedding, pose, ref.id))
        
        # Store ref #6 (the matching front-facing image)
        if ref.id == 6:
            ref_6_embedding = embedding.copy()
            ref_6_pose = pose.copy() if pose else None
            print(f"  → This is the MATCHING reference (head_front)")
        
        print()
    
    if ref_6_embedding is None or ref_6_pose is None:
        print("❌ ERROR: Reference #6 (matching image) not found or missing pose data")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Testing Pose-Aware Matching")
    print("=" * 80)
    
    # Simulate the exact matching that happened
    print(f"\nDataset Pose (same as Ref #6):")
    print(f"  yaw={ref_6_pose['yaw']:.1f}°, pitch={ref_6_pose['pitch']:.1f}°, roll={ref_6_pose['roll']:.1f}°")
    
    result_sim, best_ref_id = compute_pose_aware_similarity(
        ref_6_embedding,  # Dataset embedding (same as ref #6)
        ref_6_pose,  # Dataset pose (same as ref #6)
        reference_data,  # All 11 references
        pose_threshold=30.0
    )
    
    print(f"\n📊 Pose-Aware Similarity: {result_sim * 100:.1f}%")
    print(f"📊 Best Match: Reference #{best_ref_id}")
    
    if abs(result_sim * 100 - 69.1) < 1.0:
        print("\n✅ REPRODUCED: This matches the 69.1% score from the database!")
    else:
        print(f"\n⚠️  Score differs from expected 69.1% (got {result_sim * 100:.1f}%)")
    
    # Detailed breakdown
    print("\n" + "=" * 80)
    print("Detailed Analysis of Each Reference Contribution")
    print("=" * 80)
    
    from batch_processor import compute_similarity, compute_pose_distance
    import math
    
    sigma = 15.0
    total_weight = 0.0
    weighted_sims = []
    
    for ref_embedding, ref_pose, ref_id in reference_data:
        pose_dist = compute_pose_distance(ref_6_pose, ref_pose)
        similarity = compute_similarity(ref_6_embedding, ref_embedding)
        
        if pose_dist <= 30.0:  # Within threshold
            weight = math.exp(-pose_dist / sigma)
            total_weight += weight
            weighted_sims.append(similarity * weight)
            
            status = "✓" if pose_dist < 10.0 else "~"
            match_indicator = " ← EXACT MATCH" if ref_id == 6 else ""
            
            print(f"\n{status} Ref #{ref_id}:")
            print(f"    Similarity: {similarity * 100:.1f}%")
            print(f"    Pose Distance: {pose_dist:.1f}°")
            print(f"    Weight: {weight:.4f}")
            print(f"    Contribution: {(similarity * weight) / total_weight * 100 if total_weight > 0 else 0:.1f}%{match_indicator}")
        else:
            print(f"\n✗ Ref #{ref_id}: (excluded, pose distance {pose_dist:.1f}° > 30°)")
    
    final_score = sum(weighted_sims) / total_weight if total_weight > 0 else 0
    
    print(f"\n" + "=" * 80)
    print(f"Final Weighted Score: {final_score * 100:.1f}%")
    print("=" * 80)
    
    # Conclusion
    print("\n🔍 ROOT CAUSE:")
    if len([d for d in reference_data if compute_pose_distance(ref_6_pose, d[1]) <= 30.0]) > 1:
        print("  Multiple references are within the pose threshold (30°).")
        print("  Even though Ref #6 is an exact match (100%), it's being averaged")
        print("  with other references that have slightly different embeddings,")
        print("  causing the score to drop to ~69%.")
        print("\n💡 SOLUTION:")
        print("  1. Use best individual match instead of weighted average")
        print("  2. OR: Reduce pose threshold to avoid mixing different views")
        print("  3. OR: Give perfect matches (>99%) an absolute priority")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
