#!/usr/bin/env python3
"""
Quick test to verify reference ratios are loaded correctly after fix.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database import SessionLocal, ReferenceImage
import json
import numpy as np

db = SessionLocal()

try:
    # Load reference metrics using the FIXED code
    refs = db.query(ReferenceImage).filter(
        ReferenceImage.character_id == 1,
        ReferenceImage.body_metrics_json.isnot(None)
    ).all()
    
    reference_betas = []
    reference_ratios = []
    
    for ref in refs:
        # Load betas
        if ref.betas_blob:
            betas = np.frombuffer(ref.betas_blob, dtype=np.float32)
            reference_betas.append(betas)
        
        # Load ratios with FIXED extraction
        if ref.body_metrics_json:
            body_metrics = json.loads(ref.body_metrics_json)
            # Extract 2D ratios from nested structure
            if isinstance(body_metrics, dict) and 'ratios' in body_metrics:
                ratios_data = body_metrics['ratios']
                if 'metrics_2d' in ratios_data:
                    metrics_2d = ratios_data['metrics_2d']
                    if isinstance(metrics_2d, dict) and 'ratios' in metrics_2d:
                        reference_ratios.append(metrics_2d['ratios'])
    
    print("✅ Reference Metrics Loading Test")
    print("=" * 60)
    print(f"Loaded {len(reference_betas)} beta arrays")
    print(f"Loaded {len(reference_ratios)} ratio dictionaries")
    print()
    
    if reference_betas:
        print(f"Sample beta array shape: {reference_betas[0].shape}")
        print(f"Sample beta values: {reference_betas[0][:5]}")
        print()
    
    if reference_ratios:
        print("Sample ratio dict keys:", list(reference_ratios[0].keys()))
        print("Sample ratio values:")
        for k, v in reference_ratios[0].items():
            print(f"  {k}: {v:.3f}")
        print()
    
    # Test the comparison function
    if reference_ratios:
        from vision_engine import compute_ratio_consistency
        
        # Use first reference as test dataset image
        test_ratios = reference_ratios[0]
        consistency = compute_ratio_consistency(test_ratios, reference_ratios)
        
        print(f"Testing compute_ratio_consistency with identical ratios:")
        print(f"  Result: {consistency * 100:.1f}%")
        print(f"  Expected: ~100% (identical reference)")
        
        if consistency and consistency > 0.95:
            print("\n✅ PASS: Consistency score is correct!")
        else:
            print(f"\n❌ FAIL: Expected ~100%, got {consistency * 100:.1f}%")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
