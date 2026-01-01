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


async def process_batch(character_id: int, job_id: str):
    """
    Process all pending images for a character.
    
    This runs in a background task and updates the job progress.
    Uses serial processing (one at a time) to manage VRAM.
    """
    from vision_engine import ModelManager, FaceAnalyzer, PoseEstimator, BodyAnalyzer
    
    # Initialize cancellation flag
    cancellation_flags[job_id] = False
    
    db = SessionLocal()
    try:
        # Get job and character
        job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        character = db.query(Character).filter(Character.id == character_id).first()
        
        if not job or not character:
            logger.error(f"Job {job_id} or character {character_id} not found")
            return
        
        # Update job status
        job.status = JobStatus.PROCESSING
        db.commit()
        
        # Get pending images
        pending_images = db.query(DatasetImage).filter(
            DatasetImage.character_id == character_id,
            DatasetImage.status == ImageStatus.PENDING
        ).all()
        
        job.total_images = len(pending_images)
        db.commit()
        
        # Get master embedding for face comparison
        master_embedding = get_master_embedding(db, character_id)
        
        # Initialize models
        manager = ModelManager()
        face_analyzer = FaceAnalyzer(manager)
        pose_estimator = PoseEstimator(manager)
        body_analyzer = BodyAnalyzer(manager)
        
        # Process each image serially
        for i, dataset_image in enumerate(pending_images):
            # Check for cancellation
            if is_cancelled(job_id):
                job.status = JobStatus.CANCELLED
                job.cancelled = True
                db.commit()
                await broadcast_progress(job_id, {
                    "type": "cancelled",
                    "processed": i,
                    "total": len(pending_images)
                })
                break
            
            try:
                image_path = dataset_image.original_path
                
                # Broadcast current progress
                await broadcast_progress(job_id, {
                    "type": "progress",
                    "processed": i,
                    "total": len(pending_images),
                    "current_image": Path(image_path).name,
                    "current_path": image_path
                })
                
                # Analyze image
                face_result = face_analyzer.analyze(image_path)
                pose_result = pose_estimator.estimate(image_path)
                
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
                if face_result.detected and face_result.embedding is not None and master_embedding is not None:
                    face_similarity = compute_similarity(master_embedding, face_result.embedding)
                
                # Body analysis for full/medium shots
                body_consistency = None
                limb_ratios = None
                if shot_type in ["full-body", "medium"]:
                    body_result = body_analyzer.analyze(
                        image_path,
                        keypoints=pose_result.keypoints if pose_result.detected else None,
                        gender=character.gender.value if character.gender else "neutral"
                    )
                    if body_result.analyzed and body_result.ratios:
                        limb_ratios = body_result.ratios
                        # Simple consistency score based on ratio validity
                        body_consistency = 0.8  # Placeholder - implement proper comparison
                
                # Create or update metrics
                metrics = db.query(ImageMetrics).filter(
                    ImageMetrics.image_id == dataset_image.id
                ).first()
                
                if not metrics:
                    metrics = ImageMetrics(image_id=dataset_image.id)
                    db.add(metrics)
                
                metrics.face_similarity_score = face_similarity
                metrics.body_consistency_score = body_consistency
                metrics.shot_type = shot_type
                if limb_ratios:
                    import json
                    metrics.limb_ratios_json = json.dumps(limb_ratios)
                
                # Update image status based on scores
                if face_similarity is not None:
                    if face_similarity >= 0.85:
                        dataset_image.status = ImageStatus.APPROVED
                    elif face_similarity < 0.7:
                        dataset_image.status = ImageStatus.REJECTED
                    else:
                        dataset_image.status = ImageStatus.ANALYZED
                else:
                    dataset_image.status = ImageStatus.ANALYZED
                
                # Update job progress
                job.processed_count = i + 1
                db.commit()
                
                # Small delay to prevent UI flooding
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.exception(f"Error processing image {dataset_image.id}: {e}")
                dataset_image.status = ImageStatus.ANALYZED  # Mark as analyzed even with errors
                db.commit()
        
        # Final status update
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
        
        await broadcast_progress(job_id, {
            "type": "error",
            "message": str(e)
        })
    
    finally:
        db.close()
        # Cleanup cancellation flag
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
