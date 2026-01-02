# Guide: Implementing Optimization-Based SMPL Fitting for Body Consistency

## Overview

This document provides a complete guide for implementing proper, deterministic 3D body consistency matching using optimization-based SMPL fitting instead of the current regression-based approach.

## Current Problem

**SMPLest-X (Current)**:
- Neural network that predicts SMPL β parameters from image pixels
- Non-deterministic: Same image → different β values each run
- Example: Reference image β vs same image in dataset had L2 distance of 3.297
- Unsuitable for identity consistency (needs determinism)

**Optimization-Based Fitting (Needed)**:
- Takes deterministic 2D keypoints as input (from MMPose)
- Optimizes SMPL parameters to fit keypoints via gradient descent
- Deterministic: Same keypoints → same β values
- Industry standard for multi-view consistency

## System Architecture

### Phase 1: Reference Processing
```
Reference Images → MMPose (2D keypoints) → SMPL Optimization → Shared β
                                                                    ↓
                                                            Store in database
```

### Phase 2: Dataset Processing  
```
Dataset Image → MMPose (2D keypoints) → Evaluate fit against reference β
                                              ↓
                                    Compute reprojection error
                                              ↓
                                    Convert to consistency score
```

## Implementation Steps

### 1. Add SMPL Model Loading

**Required**:
- SMPL body model files (neutral, male, female)
- PyTorch for optimization
- Already have: MMPose keypoints, gender metadata

**Code Location**: [vision_engine.py](file:///Users/johan/Development/archetype/backend/vision_engine.py) - ModelManager class

```python
class ModelManager:
    def load_smpl_model(self, gender="neutral"):
        """Load SMPL body model for optimization."""
        import smplx
        model_path = Path(__file__).parent / "models" / "smpl"
        
        self._smpl_model = smplx.create(
            model_path=str(model_path),
            model_type='smpl',
            gender=gender,
            batch_size=1
        )
        return True
```

### 2. Implement Joint Regressor

SMPL model outputs 3D vertices. We need to convert to 2D keypoints for comparison with MMPose detections.

**Code Location**: New file `smpl_optimizer.py`

```python
import torch
import numpy as np

class SMPLOptimizer:
    def __init__(self, smpl_model, joint_regressor):
        self.smpl_model = smpl_model
        self.joint_regressor = joint_regressor  # Matrix to convert vertices → joints
        
    def vertices_to_joints(self, vertices):
        """Convert SMPL vertices to joint locations."""
        # J = joint_regressor @ vertices
        joints_3d = torch.matmul(self.joint_regressor, vertices)
        return joints_3d
    
    def project_to_2d(self, joints_3d, camera):
        """Project 3D joints to 2D using weak perspective camera."""
        # Simple orthographic projection + scale + translation
        joints_2d = camera['scale'] * joints_3d[:, :, :2] + camera['translation']
        return joints_2d
```

### 3. Optimize β for Reference Images

**Goal**: Find a SINGLE β that fits ALL reference body images

```python
def optimize_shared_beta(keypoints_list, smpl_model):
    """
    Optimize a shared β parameter across multiple views.
    
    Args:
        keypoints_list: List of dict with 'keypoints' and 'confidence'
                       from MMPose for each reference image
        smpl_model: SMPL model instance
    
    Returns:
        Optimized β parameters (shape: (10,))
    """
    # Initialize parameters
    beta = torch.zeros(1, 10, requires_grad=True)
    pose = torch.zeros(1, 72, requires_grad=False)  # T-pose
    cameras = [{'scale': torch.tensor([1.0]), 
                'translation': torch.zeros(2)} 
               for _ in keypoints_list]
    
    optimizer = torch.optim.Adam([beta], lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        smpl_output = smpl_model(betas=beta, body_pose=pose)
        vertices = smpl_output.vertices
        
        total_loss = 0
        for kpts_data, camera in zip(keypoints_list, cameras):
            # Get 3D joints from vertices
            joints_3d = vertices_to_joints(vertices)
            
            # Project to 2D
            joints_2d_pred = project_to_2d(joints_3d, camera)
            
            # Compute reprojection error (weighted by confidence)
            kpts_gt = torch.tensor(kpts_data['keypoints'])
            confidence = torch.tensor(kpts_data['confidence'])
            
            error = (joints_2d_pred - kpts_gt) ** 2
            loss = (error * confidence.unsqueeze(-1)).sum()
            total_loss += loss
        
        total_loss.backward()
        optimizer.step()
    
    return beta.detach().numpy()[0]
```

### 4. Evaluate Dataset Images

**Goal**: Measure how well dataset image keypoints fit the reference β

```python
def compute_fitting_consistency(dataset_keypoints, reference_beta, smpl_model):
    """
    Compute consistency by measuring keypoint fitting error.
    
    Args:
        dataset_keypoints: MMPose keypoints from dataset image
        reference_beta: Optimized β from references
        smpl_model: SMPL model instance
    
    Returns:
        Consistency score 0-1 (1 = perfect fit)
    """
    # Fix beta to reference value
    beta = torch.tensor(reference_beta).unsqueeze(0)
    pose = torch.zeros(1, 72)
    
    # Optimize only camera parameters to fit dataset keypoints
    camera_scale = torch.tensor([1.0], requires_grad=True)
    camera_translation = torch.zeros(2, requires_grad=True)
    
    optimizer = torch.optim.Adam([camera_scale, camera_translation], lr=0.01)
    
    for _ in range(50):
        optimizer.zero_grad()
        
        smpl_output = smpl_model(betas=beta, body_pose=pose)
        joints_3d = vertices_to_joints(smpl_output.vertices)
        joints_2d_pred = camera_scale * joints_3d[:, :, :2] + camera_translation
        
        kpts_gt = torch.tensor(dataset_keypoints['keypoints'])
        confidence = torch.tensor(dataset_keypoints['confidence'])
        
        error = ((joints_2d_pred - kpts_gt) ** 2 * confidence.unsqueeze(-1)).sum()
        error.backward()
        optimizer.step()
    
    # Final reprojection error (normalized)
    final_error = error.item() / len(kpts_gt)
    
    # Convert to similarity score (exponential decay)
    # Lower error = higher similarity
    consistency = np.exp(-final_error / threshold)
    return consistency
```

### 5. Database Schema Updates

**Add to [ReferenceImage](file:///Users/johan/Development/archetype/backend/database.py#127-144) table**:
```python
optimized_beta_blob = Column(LargeBinary)  # Shared β from all body references
```

**Migration**:
```sql
ALTER TABLE reference_images ADD COLUMN optimized_beta_blob BLOB;
```

### 6. Update Reference Processing Pipeline

**Location**: [main.py](file:///Users/johan/Development/archetype/backend/main.py) - [set_reference_images()](file:///Users/johan/Development/archetype/backend/main.py#775-857) endpoint

```python
# After processing all references
if body_reference_images:
    # Collect keypoints from all body references
    keypoints_list = []
    for ref in body_reference_images:
        pose_result = pose_estimator.estimate(ref.path)
        if pose_result.detected:
            keypoints_list.append({
                'keypoints': pose_result.keypoints,
                'confidence': pose_result.confidence
            })
    
    # Optimize shared beta
    if keypoints_list:
        smpl_model = model_manager.load_smpl_model(gender)
        shared_beta = optimize_shared_beta(keypoints_list, smpl_model)
        
        # Store in ONE reference (e.g., body_front)
        body_front_ref = next(r for r in body_reference_images if r.view_type == 'body_front')
        body_front_ref.optimized_beta_blob = shared_beta.astype(np.float32).tobytes()
```

### 7. Update Dataset Processing

**Location**: [batch_processor.py](file:///Users/johan/Development/archetype/backend/batch_processor.py) - load optimized beta

```python
# Load optimized beta (not individual betas)
optimized_beta = None
for ref in references:
    if ref.optimized_beta_blob:
        optimized_beta = np.frombuffer(ref.optimized_beta_blob, dtype=np.float32)
        break  # Only need one (shared across all body refs)

# Pass to body analyzer
body_result = body_analyzer.analyze(
    image_path,
    keypoints=pose_result.keypoints,
    optimized_beta=optimized_beta  # NEW parameter
)
```

### 8. Update BodyAnalyzer

**Location**: [vision_engine.py](file:///Users/johan/Development/archetype/backend/vision_engine.py) - `BodyAnalyzer.analyze()`

```python
def analyze(self, image_path, keypoints=None, 
            optimized_beta=None):  # NEW: reference beta
    
    # ... existing code ...
    
    if optimized_beta is not None and keypoints:
        # Use optimization-based consistency
        smpl_model = self._manager.load_smpl_model(gender)
        consistency_3d = compute_fitting_consistency(
            keypoints, optimized_beta, smpl_model
        )
    else:
        # Fall back to 2D ratios only
        consistency_3d = None
```

## Expected Results

### Before (Current Broken State)
```
Reference: Hanna_3efc109e.png
Dataset: Hanna_3efc109e.png (same image)
3D Consistency: 19.2%  ❌ (due to non-deterministic betas)
```

### After (Optimization-Based)
```
Reference: Hanna_3efc109e.png
Dataset: Hanna_3efc109e.png (same image)
3D Consistency: 99.8%  ✅ (deterministic keypoint fitting)
```

## Implementation Prompt

When ready to implement, use this prompt:

---

**Prompt**:

> I need to replace the current SMPLest-X regression-based body consistency with deterministic optimization-based SMPL fitting. The current system has a critical flaw: identical images produce different β parameters (L2 distance of 3.297), resulting in only 19% consistency scores.
>
> **Requirements**:
> 1. Use SMPL body model (not SMPL-X) with PyTorch optimization
> 2. During reference processing: Optimize a SINGLE shared β across all body reference images using their MMPose keypoints
> 3. During dataset processing: Fix β to the reference value and optimize only camera parameters, measuring reprojection error as the consistency metric
> 4. Store optimized β in `ReferenceImage.optimized_beta_blob`
> 5. Update `BodyAnalyzer.analyze()` to accept `optimized_beta` parameter
> 6. Convert reprojection error to 0-1 consistency score using exponential decay
>
> **Key Files to Modify**:
> - [vision_engine.py](file:///Users/johan/Development/archetype/backend/vision_engine.py): Add `SMPLOptimizer` class, update [ModelManager](file:///Users/johan/Development/archetype/backend/vision_engine.py#299-509) and [BodyAnalyzer](file:///Users/johan/Development/archetype/backend/vision_engine.py#781-1017)
> - [main.py](file:///Users/johan/Development/archetype/backend/main.py): Update [set_reference_images()](file:///Users/johan/Development/archetype/backend/main.py#775-857) to optimize shared β
> - [batch_processor.py](file:///Users/johan/Development/archetype/backend/batch_processor.py): Load optimized β and pass to analyzer
> - [database.py](file:///Users/johan/Development/archetype/backend/database.py): Add `optimized_beta_blob` column
>
> **Expected Outcome**: 
> Same image should show ~100% 3D consistency (deterministic), not ~20% (current broken state).
>
> Please implement this following the architecture in `optimization_based_smpl_fitting.md`.

---

## Dependencies

```bash
pip install smplx torch numpy
```

Download SMPL models from: https://smpl.is.tue.mpg.de/
Place in: `backend/models/smpl/`

## References

- **SMPLify**: Original optimization-based fitting paper
- **SMPL-X**: Extended model (we use basic SMPL for simpler optimization)
- **PyTorch SMPL**: https://github.com/vchoutas/smplx

## Notes

- Optimization-based fitting is ~10-20x slower than regression (acceptable for reference processing)
- For dataset processing, we only optimize 3 camera parameters (fast)
- The shared β assumption works because all images are of the same character
- This approach is deterministic and theoretically correct for identity matching
