"""
Batch processor for image analysis with WebSocket progress updates.

Handles serial processing to manage VRAM, with cancellation support.
"""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional
import numpy as np

from sqlalchemy.orm import Session

from database import (
    SessionLocal, DatasetImage, ImageMetrics, ReferenceImage, 
    ProcessingJob, JobStatus, ImageStatus, Character
)

logger = logging.getLogger(__name__)

# Store for active WebSocket connections by job_id
active_connections: dict[str, list] = {}

# Store for cancellation flags
cancellation_flags: dict[str, bool] = {}


def register_websocket(job_id: str, websocket):
    """Register a WebSocket connection for a job."""
    if job_id not in active_connections:
        active_connections[job_id] = []
    active_connections[job_id].append(websocket)


def unregister_websocket(job_id: str, websocket):
    """Remove a WebSocket connection for a job."""
    if job_id in active_connections:
        active_connections[job_id] = [
            ws for ws in active_connections[job_id] if ws != websocket
        ]
        if not active_connections[job_id]:
            del active_connections[job_id]


async def broadcast_progress(job_id: str, data: dict):
    """Send progress update to all connected WebSockets."""
    if job_id in active_connections:
        for websocket in active_connections[job_id]:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")


def request_cancellation(job_id: str) -> bool:
    """Request cancellation of a job."""
    if job_id in cancellation_flags:
        cancellation_flags[job_id] = True
        return True
    return False


def is_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    return cancellation_flags.get(job_id, False)


def get_master_embedding(db: Session, character_id: int) -> Optional[np.ndarray]:
    """Get the master face embedding for a character from reference images."""
    ref_images = db.query(ReferenceImage).filter(
        ReferenceImage.character_id == character_id,
        ReferenceImage.embedding_blob.isnot(None)
    ).all()
    
    if not ref_images:
        return None
    
    embeddings = []
    for ref in ref_images:
        if ref.embedding_blob:
            try:
                embedding = np.frombuffer(ref.embedding_blob, dtype=np.float32)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to decode embedding for ref {ref.id}: {e}")
    
    if embeddings:
        return np.mean(embeddings, axis=0)
    return None


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def compute_pose_distance(pose1: dict, pose2: dict) -> float:
    """
    Compute angular distance between two head poses.
    
    Args:
        pose1, pose2: Dicts with keys 'yaw', 'pitch', 'roll' (in degrees)
    
    Returns:
        Distance in degrees (weighted Euclidean)
    """
    import math
    
    yaw_diff = abs(pose1['yaw'] - pose2['yaw'])
    pitch_diff = abs(pose1['pitch'] - pose2['pitch'])
    roll_diff = abs(pose1['roll'] - pose2['roll'])
    
    # Weighted Euclidean distance (yaw is most important)
    distance = math.sqrt(
        (yaw_diff * 2.0) ** 2 +   # Yaw weight = 2x (left/right is critical)
        (pitch_diff * 1.0) ** 2 +  # Pitch weight = 1x  
        (roll_diff * 0.5) ** 2     # Roll weight = 0.5x (tilt less important)
    )
    return distance


def compute_pose_aware_similarity(
    dataset_embedding: np.ndarray,
    dataset_pose: dict,
    reference_data: list[tuple[np.ndarray, dict, int]],  # (embedding, pose, ref_id)
    pose_threshold: float = 30.0  # degrees
) -> tuple[float, Optional[int]]:
    """
    Compute face similarity with pose-aware weighting.
    
    Compares dataset image against references with similar head poses,
    weighting by inverse pose distance.
    
    Args:
        dataset_embedding: Face embedding of dataset image
        dataset_pose: Head pose of dataset image
        reference_data: List of (embedding, pose, ref_id) tuples from references
        pose_threshold: Max pose distance to consider (degrees)
    
    Returns:
        Tuple of (Weighted similarity score, Best match reference ID)
    """
    import math
    
    weighted_similarities = []
    total_weight = 0.0
    
    best_score = -1.0
    best_ref_id = None
    
    for ref_embedding, ref_pose, ref_id in reference_data:
        # Calculate pose distance
        pose_dist = compute_pose_distance(dataset_pose, ref_pose)
        
        # Skip references too far in pose space
        if pose_dist > pose_threshold:
            continue
        
        # Calculate embedding similarity
        similarity = compute_similarity(dataset_embedding, ref_embedding)
        
        # Track best individual match
        if similarity > best_score:
            best_score = similarity
            best_ref_id = ref_id
        
        # PRIORITY: If we found a near-perfect match (>95%), use it directly
        # This prevents dilution from averaging with lower-scoring references
        if similarity >= 0.95:
            return (similarity, ref_id)
        
        # Weight by inverse pose distance (exponential decay)
        # weight = exp(-distance / sigma)
        sigma = 15.0  # Controls decay rate
        weight = math.exp(-pose_dist / sigma)
        
        weighted_similarities.append(similarity * weight)
        total_weight += weight
    
    if total_weight == 0:
        # No close references found - fall back to best match across all
        best_score = -1.0
        best_ref_id = None
        
        for ref_emb, ref_pose, ref_id in reference_data:
            sim = compute_similarity(dataset_embedding, ref_emb)
            if sim > best_score:
                best_score = sim
                best_ref_id = ref_id
                
        return (best_score, best_ref_id) if best_score >= 0 else (0.0, None)
    
    # Use best individual match instead of weighted average
    # This fixes the issue where perfect matches were being diluted
    return best_score, best_ref_id


async def process_single_dataset_image(
    db: Session,
    dataset_image: DatasetImage,
    reference_data: list,
    models: dict,
    master_embedding: Optional[np.ndarray] = None
) -> dict:
    """
    Process a single dataset image and update its metrics.
    
    Args:
        db: Database session
        dataset_image: Image to process
        reference_data: List of (embedding, pose, ref_id) tuples
        models: Dictionary containing initialized analyzer instances
        master_embedding: Fallback embedding for non-pose-aware matching
        
    Returns:
        Dictionary with processing results/metrics
    """
    from database import ImageStatus
    
    # Unpack models
    face_analyzer = models['face_analyzer']
    pose_estimator = models['pose_estimator']
    body_analyzer = models['body_analyzer']
    
    image_path = dataset_image.original_path
    character = dataset_image.character
    
    # Run analysis tasks
    loop = asyncio.get_event_loop()
    
    def _vision_task():
        logger.info(f"Running Face Analysis for {Path(image_path).name}...")
        f_res = face_analyzer.analyze(image_path)
        logger.info(f"Running Pose Estimation for {Path(image_path).name}...")
        p_res = pose_estimator.estimate(image_path)
        return f_res, p_res

    face_result, pose_result = await loop.run_in_executor(None, _vision_task)
    
    # Determine shot type
    shot_type = "unknown"
    if pose_result.detected and pose_result.keypoints:
        kp = pose_result.keypoints
        has_full_body = all(
            k in kp and kp[k].get('confidence', 0) > 0.3
            for k in ['left_ankle', 'right_ankle', 'left_shoulder', 'right_shoulder']
        )
        has_upper = all(
            k in kp and kp[k].get('confidence', 0) > 0.3
            for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        )
        if has_full_body:
            shot_type = "full-body"
        elif has_upper:
            shot_type = "medium"
        elif face_result.detected:
            shot_type = "close-up"
    elif face_result.detected:
        shot_type = "close-up"
    
    # Compute face similarity
    face_similarity = None
    closest_face_ref_id = None
    
    if face_result.detected and face_result.embedding is not None:
        if face_result.pose and reference_data:
            # Use pose-aware similarity
            face_similarity, closest_face_ref_id = compute_pose_aware_similarity(
                face_result.embedding,
                face_result.pose,
                reference_data
            )
        elif master_embedding is not None:
            # Fallback
            face_similarity = compute_similarity(master_embedding, face_result.embedding)
    
    # Body analysis
    body_consistency = None
    body_consistency_3d = None
    body_consistency_2d = None
    limb_ratios = None
    
    if shot_type in ["full-body", "medium"]:
        def _body_task():
            logger.info(f"Running Body Analysis for {Path(image_path).name}...")
            return body_analyzer.analyze(
                image_path,
                keypoints=pose_result.keypoints if pose_result.detected else None,
                gender=character.gender.value if character.gender else "neutral"
            )
        
        body_result = await loop.run_in_executor(None, _body_task)
        
        if body_result.analyzed and body_result.ratios:
            metrics_3d = body_result.ratios.get('metrics_3d')
            metrics_2d = body_result.ratios.get('metrics_2d')
            preferred = body_result.ratios.get('preferred', 'none')
            
            if metrics_3d:
                body_consistency_3d = metrics_3d.get('consistency_score')
            if metrics_2d:
                body_consistency_2d = metrics_2d.get('consistency_score')
            
            if preferred == "3d" and body_consistency_3d:
                body_consistency = body_consistency_3d
            elif preferred == "2d" and body_consistency_2d:
                body_consistency = body_consistency_2d
            
            limb_ratios = {
                "metrics_3d": metrics_3d,
                "metrics_2d": metrics_2d,
                "preferred": preferred
            }
    
    # Create or update metrics
    metrics = db.query(ImageMetrics).filter(
        ImageMetrics.image_id == dataset_image.id
    ).first()
    
    if not metrics:
        metrics = ImageMetrics(image_id=dataset_image.id)
        db.add(metrics)
    
    metrics.face_similarity_score = face_similarity
    metrics.closest_face_ref_id = closest_face_ref_id
    metrics.body_consistency_score = body_consistency
    metrics.shot_type = shot_type
    
    if pose_result.detected and pose_result.keypoints:
        import json
        metrics.keypoints_json = json.dumps(pose_result.keypoints)
    
    if face_result.detected and face_result.bbox:
        import json
        metrics.face_bbox_json = json.dumps(face_result.bbox)
    
    if face_result.detected and face_result.pose:
        import json
        metrics.face_pose_json = json.dumps(face_result.pose)
    
    if limb_ratios:
        import json
        metrics.limb_ratios_json = json.dumps(limb_ratios)
    
    # Update status
    if face_similarity is not None:
        if face_similarity >= 0.85:
            dataset_image.status = ImageStatus.APPROVED
        elif face_similarity < 0.7:
            dataset_image.status = ImageStatus.REJECTED
        else:
            dataset_image.status = ImageStatus.ANALYZED
    else:
        dataset_image.status = ImageStatus.ANALYZED
        
    return {
        "face_similarity": face_similarity,
        "body_consistency": body_consistency,
        "shot_type": shot_type
    }


async def process_batch(character_id: int, job_id: str, reprocess_all: bool = False):
    """
    Process images for a character.
    """
    from database import ImageStatus
    from vision_engine import ModelManager, FaceAnalyzer, PoseEstimator, BodyAnalyzer
    
    # Initialize cancellation flag
    cancellation_flags[job_id] = False
    
    db = SessionLocal()
    try:
        # ... (setup logic remains same until loop) ...
        # Get job and character
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        character = db.query(Character).filter(Character.id == character_id).first()
        
        if not job or not character:
            logger.error(f"Job {job_id} or character {character_id} not found")
            return
        
        # Update job status
        job.status = JobStatus.PROCESSING
        db.commit()
        
        # Get images to process
        query = db.query(DatasetImage).filter(DatasetImage.character_id == character_id)
        if not reprocess_all:
            query = query.filter(DatasetImage.status == ImageStatus.PENDING)
            
        pending_images = query.all()
        
        if not pending_images:
            job.status = JobStatus.COMPLETED
            db.commit()
            await broadcast_progress(job_id, {
                "type": "complete",
                "processed": 0,
                "total": 0
            })
            return
        
        job.total_images = len(pending_images)
        db.commit()
        
        # Get reference data
        references = db.query(ReferenceImage).filter(
            ReferenceImage.character_id == character_id,
            ReferenceImage.embedding_blob.isnot(None)
        ).all()
        
        reference_data = [] 
        master_embedding = None
        
        for ref in references:
            embedding = np.frombuffer(ref.embedding_blob, dtype=np.float32)
            if ref.pose_json:
                import json
                pose = json.loads(ref.pose_json)
                reference_data.append((embedding, pose, ref.id))
            else:
                if master_embedding is None:
                    master_embedding = embedding.copy()
                else:
                    master_embedding = (master_embedding + embedding) / 2
        
        if not reference_data and master_embedding is None:
            master_embedding = get_master_embedding(db, character_id)
        
        # Initialize models
        manager = ModelManager()
        models = {
            'face_analyzer': FaceAnalyzer(manager),
            'pose_estimator': PoseEstimator(manager),
            'body_analyzer': BodyAnalyzer(manager)
        }
        
        # Process loop
        for i, dataset_image in enumerate(pending_images):
            if is_cancelled(job_id):
                job.status = JobStatus.CANCELLED
                job.cancelled = True
                db.commit()
                await broadcast_progress(job_id, {"type": "cancelled", "processed": i, "total": len(pending_images)})
                break
            
            try:
                # Progress update
                await broadcast_progress(job_id, {
                    "type": "progress",
                    "processed": i,
                    "total": len(pending_images),
                    "current_image": Path(dataset_image.original_path).name,
                    "current_path": dataset_image.original_path
                })
                
                # Process SINGLE image
                await process_single_dataset_image(
                    db,
                    dataset_image,
                    reference_data,
                    models,
                    master_embedding
                )
                
                # Update job count
                job.processed_count = i + 1
                db.commit()
                
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.exception(f"Error processing image {dataset_image.id}: {e}")
                dataset_image.status = ImageStatus.ANALYZED
                db.commit()
        
        # Final status
        if not is_cancelled(job_id):
            job.status = JobStatus.COMPLETED
            job.processed_count = len(pending_images)
            db.commit()
            await broadcast_progress(job_id, {
                "type": "completed",
                "processed": len(pending_images),
                "total": len(pending_images)
            })
            
    except Exception as e:
        logger.exception(f"Batch processing failed: {e}")
        try:
            job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                db.commit()
        except:
            pass
        await broadcast_progress(job_id, {"type": "error", "message": str(e)})
    finally:
        db.close()
        if job_id in cancellation_flags:
            del cancellation_flags[job_id]


def scan_folder_for_images(folder_path: str) -> list[str]:
    """
    Scan a folder for supported image files.
    
    Returns list of absolute paths to found images.
    """
    supported_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder_path}")
    
    images = []
    for ext in supported_extensions:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    
    return [str(img.resolve()) for img in sorted(images)]


def create_dataset_entries(db: Session, character_id: int, image_paths: list[str]) -> int:
    """
    Create DatasetImage entries for a list of image paths.
    
    Skips paths that already exist for the character.
    Returns the number of new entries created.
    """
    existing = set(
        path for (path,) in db.query(DatasetImage.original_path).filter(
            DatasetImage.character_id == character_id
        ).all()
    )
    
    new_count = 0
    for path in image_paths:
        if path not in existing:
            entry = DatasetImage(
                character_id=character_id,
                original_path=path,
                status=ImageStatus.PENDING
            )
            db.add(entry)
            new_count += 1
    
    db.commit()
    return new_count
