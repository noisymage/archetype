#!/usr/bin/env python3
"""
Test if a single image processing actually receives reference metrics.
This simulates what happens when user clicks "Reprocess Image".
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database import SessionLocal, DatasetImage, ReferenceImage, Character
from batch_processor import process_single_dataset_image
from vision_engine import ModelManager, FaceAnalyzer, PoseEstimator, BodyAnalyzer
import numpy as np
import json

async def test_reprocess():
    db = SessionLocal()
    
    try:
        # Get a full-body image
        image = db.query(DatasetImage).join(DatasetImage.metrics).filter(
            DatasetImage.character_id == 1
        ).filter(
            DatasetImage.metrics.has(shot_type='full-body')
        ).first()
        
        if not image:
            print("❌ No full-body images found")
            return
        
        print(f"Testing with: {Path(image.original_path).name}")
        print()
        
        # Load references - EXACTLY like reprocess_image does
        character_id = image.character_id
        references = db.query(ReferenceImage).filter(
            ReferenceImage.character_id == character_id,
            ReferenceImage.embedding_blob.isnot(None)
        ).all()
        
        reference_data = []
        master_embedding = None
        reference_betas = []
        reference_ratios = []
        
        for ref in references:
            embedding = np.frombuffer(ref.embedding_blob, dtype=np.float32)
            if ref.pose_json:
                pose = json.loads(ref.pose_json)
                reference_data.append((embedding, pose, ref.id))
            else:
                if master_embedding is None:
                    master_embedding = embedding.copy()
                else:
                    master_embedding = (master_embedding + embedding) / 2
            
            # Load body metrics
            if ref.betas_blob:
                betas = np.frombuffer(ref.betas_blob, dtype=np.float32)
                reference_betas.append(betas)
            
            if ref.body_metrics_json:
                body_metrics = json.loads(ref.body_metrics_json)
                # Extract 2D ratios from nested structure
                if isinstance(body_metrics, dict) and 'ratios' in body_metrics:
                    ratios_data = body_metrics['ratios']
                    if 'metrics_2d' in ratios_data:
                        metrics_2d = ratios_data['metrics_2d']
                        if isinstance(metrics_2d, dict) and 'ratios' in metrics_2d:
                            reference_ratios.append(metrics_2d['ratios'])
        
        print(f"Loaded {len(reference_betas)} beta arrays")
        print(f"Loaded {len(reference_ratios)} ratio dicts")
        print()
        
        if not reference_ratios:
            print("❌ FAIL: No reference ratios loaded!")
            return
        
        print("✅ Reference ratios loaded successfully")
        print(f"Sample keys: {list(reference_ratios[0].keys())}")
        print()
        
        # Initialize models
        manager = ModelManager()
        models = {
            'face_analyzer': FaceAnalyzer(manager),
            'pose_estimator': PoseEstimator(manager),
            'body_analyzer': BodyAnalyzer(manager)
        }
        
        print("Processing image...")
        await process_single_dataset_image(
            db,
            image,
            reference_data,
            models,
            master_embedding,
            reference_betas,
            reference_ratios  # ← THIS IS THE KEY
        )
        
        print("Processing complete")
        
        # Check result
        db.refresh(image)
        if image.metrics:
            print()
            print(f"Result body_consistency_score: {image.metrics.body_consistency_score}")
            if image.metrics.body_consistency_score:
                print(f"✅ SUCCESS: Got {image.metrics.body_consistency_score * 100:.1f}%")
            else:
                print("❌ FAIL: body_consistency_score is still None")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_reprocess())
