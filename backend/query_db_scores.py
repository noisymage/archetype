#!/usr/bin/env python3
"""
Query database to find the 69.1% score and understand where it comes from.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database import SessionLocal, DatasetImage, ImageMetrics, ReferenceImage, Character
from sqlalchemy import desc

db = SessionLocal()

try:
    # Find images with approximately 69% similarity
    print("=" * 80)
    print("Searching for 69.1% face similarity score in database...")
    print("=" * 80)
    
    results = db.query(DatasetImage, ImageMetrics).join(ImageMetrics).filter(
        ImageMetrics.face_similarity_score.between(0.68, 0.70)
    ).all()
    
    if not results:
        print("\n⚠️  No images found with face similarity between 68-70%")
        print("\nShowing all recent analyzed images instead:\n")
        results = db.query(DatasetImage, ImageMetrics).join(ImageMetrics).filter(
            ImageMetrics.face_similarity_score.isnot(None)
        ).order_by(desc(DatasetImage.id)).limit(10).all()
    
    for dataset_img, metrics in results:
        print(f"\nImage: {Path(dataset_img.original_path).name}")
        print(f"  Full Path: {dataset_img.original_path}")
        print(f"  Character ID: {dataset_img.character_id}")
        print(f"  Status: {dataset_img.status.value}")
        print(f"  Face Similarity: {metrics.face_similarity_score * 100:.1f}%" if metrics.face_similarity_score else "  Face Similarity: None")
        print(f"  Body Consistency: {metrics.body_consistency_score * 100:.1f}%" if metrics.body_consistency_score else "  Body Consistency: None")
        print(f"  Shot Type: {metrics.shot_type}")
        print(f"  Closest Face Ref ID: {metrics.closest_face_ref_id}")
        
        # Get character name
        char = db.query(Character).filter(Character.id == dataset_img.character_id).first()
        if char:
            print(f"  Character: {char.name}")
            
        # Get reference image details if available
        if metrics.closest_face_ref_id:
            ref = db.query(ReferenceImage).filter(ReferenceImage.id == metrics.closest_face_ref_id).first()
            if ref:
                print(f"  Matched to reference: {Path(ref.path).name} ({ref.view_type})")
        
        print("  " + "-" * 76)
    
    # Check reference images for the character
    print("\n" + "=" * 80)
    print("Reference Images in Database:")
    print("=" * 80)
    
    refs = db.query(ReferenceImage, Character).join(Character).all()
    for ref, char in refs:
        has_embedding = ref.embedding_blob is not None and len(ref.embedding_blob) > 0
        has_pose = ref.pose_json is not None
        print(f"\n{char.name} - {ref.view_type}: {Path(ref.path).name}")
        print(f"  Embedding: {'✓' if has_embedding else '✗'}")
        print(f"  Pose: {'✓' if has_pose else '✗'}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
