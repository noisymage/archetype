"""
FastAPI application entry point for Character Consistency Validator.
"""
import os
import logging
import asyncio
import uuid
import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import (
    init_db, get_db, migrate_db, Project, Character, ReferenceImage, DatasetImage, 
    ImageMetrics, ProcessingJob, JobStatus, ImageStatus, LoraPresetType, Gender
)
from fastapi import UploadFile, File
import shutil
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === Pydantic Models ===

class ReferenceAnalysisRequest(BaseModel):
    """Request for reference image validation."""
    images: dict[str, str] = Field(
        ...,
        description="Mapping of view type to absolute image path",
        examples=[{
            "head_front": "/path/to/head_front.jpg",
            "head_45l": "/path/to/head_45l.png",
            "head_45r": "/path/to/head_45r.jpg",
            "body_front": "/path/to/body_front.jpg",
            "body_side": "/path/to/body_side.jpg",
            "body_posterior": "/path/to/body_posterior.jpg"
        }]
    )
    gender: str = Field(
        default="neutral",
        description="Character gender: male, female, or neutral"
    )


class SingleImageRequest(BaseModel):
    """Request for single image analysis."""
    image_path: str = Field(..., description="Absolute path to image file")


class ValidationWarningResponse(BaseModel):
    """Validation warning in response."""
    code: str
    message: str
    severity: str


class ReferenceAnalysisResponse(BaseModel):
    """Response from reference validation."""
    success: bool
    degraded_mode: bool = True
    warnings: list[ValidationWarningResponse] = []
    master_embedding: Optional[list[float]] = None
    body_metrics: Optional[dict] = None
    error: Optional[str] = None


class SingleImageResponse(BaseModel):
    """Response from single image analysis."""
    face_detected: bool = False
    face_bbox: Optional[list[float]] = None
    body_detected: bool = False
    keypoints: Optional[dict] = None
    shot_type: str = "unknown"
    error: Optional[str] = None


class DetailedAnalysisResponse(BaseModel):
    """Full analysis response for the tool."""
    face: Optional[dict] = None
    pose: Optional[dict] = None
    body: Optional[dict] = None
    error: Optional[str] = None


# --- Project/Character CRUD Models ---

class CharacterCreate(BaseModel):
    """Request to create a character."""
    name: str = Field(..., min_length=1, max_length=255)
    gender: str = Field(default="neutral")


class CharacterResponse(BaseModel):
    """Character data response."""
    id: int
    project_id: int
    name: str
    gender: Gender
    reference_count: int = 0
    image_count: int = 0
    reference_images_path: Optional[str] = None
    dataset_images_path: Optional[str] = None

    class Config:
        from_attributes = True


# ... Project models ... (assumed unchanged from previous valid state or I'll just touch CharacterResponse)

# ...




class ProjectCreate(BaseModel):
    """Request to create a project."""
    name: str = Field(..., min_length=1, max_length=255)
    lora_preset_type: str = Field(default="SDXL")


class ProjectResponse(BaseModel):
    """Project data response."""
    id: int
    name: str
    lora_preset_type: LoraPresetType
    character_count: int = 0
    characters: list[CharacterResponse]

    class Config:
        from_attributes = True


class ReferenceImageCreate(BaseModel):
    """Request to add reference images."""
    images: dict[str, str] = Field(
        ...,
        description="Mapping of view type to absolute image path"
    )
    source_path: Optional[str] = None


class DatasetImageResponse(BaseModel):
    """Dataset image data response."""
    id: int
    original_path: str
    filename: str
    status: str
    face_similarity: Optional[float] = None
    body_consistency: Optional[float] = None
    shot_type: Optional[str] = None
    limb_ratios: Optional[dict] = None
    keypoints: Optional[dict] = None
    face_bbox: Optional[list] = None
    closest_face_ref: Optional[str] = None  # Path to closest reference image

    class Config:
        from_attributes = True


class FolderScanRequest(BaseModel):
    """Request to scan a folder for images."""
    folder_path: str = Field(..., description="Absolute path to folder")
    character_id: int


class FolderScanResponse(BaseModel):
    """Response from folder scan."""
    folder_path: str
    total_found: int
    new_entries: int
    already_exists: int


class BatchProcessRequest(BaseModel):
    """Request to start batch processing."""
    character_id: int
    reprocess_all: bool = False


class BatchProcessResponse(BaseModel):
    """Response from batch process start."""
    job_id: str
    character_id: int
    total_images: int
    status: str


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    total_images: int
    processed_count: int
    error_message: Optional[str] = None


# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - initialize database on startup."""
    init_db()
    migrate_db()  # Run migrations
    # Ensure models directory exists
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "insightface").mkdir(exist_ok=True)
    (models_dir / "smplx" / "body_models").mkdir(parents=True, exist_ok=True)
    (models_dir / "smplx" / "pretrained").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Archetype - Character Consistency Validator",
    description="API for validating and curating datasets for LoRA training",
    version="0.2.0",
    lifespan=lifespan
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Utility Functions ===

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg'}


def validate_image_path(path: str) -> tuple[bool, str]:
    """
    Validate an image path for security and existence.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Security: prevent directory traversal
    try:
        resolved = Path(path).resolve()
        # Basic security check - path should not contain suspicious patterns
        path_str = str(resolved)
        if '..' in path or not resolved.is_absolute():
            return False, "Invalid path: directory traversal not allowed"
    except Exception as e:
        return False, f"Invalid path: {str(e)}"
    
    # Check existence
    if not resolved.exists():
        return False, f"File not found: {path}"
    
    # Check format
    if resolved.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        return False, f"Unsupported format. Supported: {SUPPORTED_IMAGE_FORMATS}"
    
    return True, ""


def _convert_numpy(obj):
    """Recursively convert numpy objects to python built-ins."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    return obj


# === Processing Endpoints ===

@app.post("/api/images/{image_id}/reprocess", response_model=DatasetImageResponse)
async def reprocess_image(image_id: int, db: Session = Depends(get_db)):
    """Reprocess a single image."""
    image = db.query(DatasetImage).filter(DatasetImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    from batch_processor import process_single_dataset_image, get_master_embedding
    from vision_engine import ModelManager, FaceAnalyzer, PoseEstimator, BodyAnalyzer
    import numpy as np

    # Prepare reference data
    character_id = image.character_id
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

    # Init models
    manager = ModelManager()
    models = {
        'face_analyzer': FaceAnalyzer(manager),
        'pose_estimator': PoseEstimator(manager),
        'body_analyzer': BodyAnalyzer(manager)
    }

    try:
        await process_single_dataset_image(
            db,
            image,
            reference_data,
            models,
            master_embedding
        )
        db.commit()
        db.refresh(image)
        
        # Format response
        metrics = image.metrics
        closest_ref_path = None
        if metrics and metrics.closest_face_ref:
            closest_ref_path = metrics.closest_face_ref.path
            
        return DatasetImageResponse(
            id=image.id,
            original_path=image.original_path,
            filename=Path(image.original_path).name,
            status=image.status.value if image.status else "pending",
            face_similarity=metrics.face_similarity_score if metrics else None,
            body_consistency=metrics.body_consistency_score if metrics else None,
            shot_type=metrics.shot_type if metrics else None,
            limb_ratios=json.loads(metrics.limb_ratios_json) if metrics and metrics.limb_ratios_json else None,
            keypoints=json.loads(metrics.keypoints_json) if metrics and metrics.keypoints_json else None,
            face_bbox=json.loads(metrics.face_bbox_json) if metrics and metrics.face_bbox_json else None,
            closest_face_ref=closest_ref_path
        )
        
    except Exception as e:
        logger.exception(f"Reprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === API Endpoints ===

@app.get("/api/health")
async def health_check():
    """Health check endpoint for frontend connectivity verification."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Archetype API - Character Consistency Validator"}


@app.post("/api/analyze/reference", response_model=ReferenceAnalysisResponse)
async def analyze_reference(request: ReferenceAnalysisRequest):
    """
    Validate a set of 6 reference images for consistency.
    
    Runs face analysis on head group, pose/body analysis on body group,
    and performs cross-validation checks.
    """
    try:
        # Validate all image paths
        for view_type, path in request.images.items():
            is_valid, error = validate_image_path(path)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid path for {view_type}: {error}"
                )
        
        # Import vision engine (lazy import to avoid startup overhead)
        from vision_engine import validate_references
        
        # Run validation
        result = validate_references(request.images, gender=request.gender)
        
        # Convert to response
        return ReferenceAnalysisResponse(
            success=result.success,
            degraded_mode=result.degraded_mode,
            warnings=[
                ValidationWarningResponse(
                    code=w.code,
                    message=w.message,
                    severity=w.severity
                )
                for w in result.warnings
            ],
            master_embedding=result.master_embedding.tolist() if result.master_embedding is not None else None,
            body_metrics=_convert_numpy(result.body_metrics),
            error=result.error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Reference analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/image", response_model=SingleImageResponse)
async def analyze_image(request: SingleImageRequest):
    """
    Analyze a single image for face, body, and shot type.
    
    Returns bounding boxes, keypoints, and detected shot type
    (close-up, medium, or full-body).
    """
    try:
        # Validate path
        is_valid, error = validate_image_path(request.image_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Import vision engine (lazy import)
        from vision_engine import analyze_single_image
        
        # Run analysis
        result = analyze_single_image(request.image_path)
        
        return SingleImageResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


        raise HTTPException(status_code=500, detail=str(e))


def _run_detailed_analysis(image_path: str) -> DetailedAnalysisResponse:
    """Shared logic for detailed analysis."""
    from vision_engine import ModelManager, FaceAnalyzer, PoseEstimator, BodyAnalyzer
    
    manager = ModelManager()
    
    # Face
    face_analyzer = FaceAnalyzer(manager)
    face_result = face_analyzer.analyze(image_path)
    
    # Pose
    pose_estimator = PoseEstimator(manager)
    pose_result = pose_estimator.estimate(image_path)
    
    # Body (needs keypoints from pose if available)
    body_analyzer = BodyAnalyzer(manager)
    keypoints = pose_result.keypoints if pose_result.detected else None
    bbox = pose_result.bbox if pose_result.detected else None
    
    # Pass bbox to skip redundant detection
    body_result = body_analyzer.analyze(image_path, keypoints=keypoints, bbox=bbox)
    
    # Construct response
    resp = DetailedAnalysisResponse()
    
    if face_result.detected:
        resp.face = {
            "bbox": face_result.bbox,
            "confidence": face_result.confidence,
            "landmarks": face_result.landmarks.tolist() if face_result.landmarks is not None else None,
            "embedding": face_result.embedding.tolist() if face_result.embedding is not None else None,
            "pose": face_result.pose
        }
    elif face_result.error:
        resp.face = {"error": face_result.error}
        
    if pose_result.detected:
        resp.pose = {
            "keypoints": pose_result.keypoints,
            "bbox": pose_result.bbox,
            "confidence": pose_result.confidence
        }
    elif pose_result.error:
            resp.pose = {"error": pose_result.error}
            
    if body_result.analyzed:
            resp.body = {
                "degraded_mode": body_result.degraded_mode,
                "betas": body_result.betas.tolist() if body_result.betas is not None else None,
                "volume_estimate": body_result.volume_estimate,
                "ratios": body_result.ratios
            }
    elif body_result.error:
            resp.body = {"error": body_result.error}
            
    return resp


@app.post("/api/tools/analyze_detailed", response_model=DetailedAnalysisResponse)
async def analyze_detailed(request: SingleImageRequest):
    """
    Detailed analysis for comparison tool (path based).
    """
    try:
        # Validate path
        is_valid, error = validate_image_path(request.image_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
            
        return _run_detailed_analysis(request.image_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Detailed analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/analyze_upload", response_model=DetailedAnalysisResponse)
async def analyze_upload(file: UploadFile = File(...)):
    """
    Detailed analysis for comparison tool (upload based).
    """
    try:
        # Create temp file
        suffix = Path(file.filename).suffix
        if suffix not in SUPPORTED_IMAGE_FORMATS:
             # Try to guess or just allow? Better to be strict or default to .jpg
             if not suffix: suffix = ".jpg"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            return _run_detailed_analysis(tmp_path)
        finally:
            # Cleanup temp file? 
            # If we want to show the image in frontend, we might need to serve it.
            # But frontend has the file already (HTML5 File object).
            # So we can just return metrics.
            # We can delete the temp file.
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        logger.exception("Upload analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image/serve")
async def serve_image(path: str = Query(..., description="Absolute path to image file")):
    """
    Serve an image file by its absolute path.
    
    Security: Validates path and only serves supported image formats.
    """
    # Validate path
    is_valid, error = validate_image_path(path)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    resolved = Path(path).resolve()
    
    # Determine media type
    suffix = resolved.suffix.lower()
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg'
    }
    media_type = media_types.get(suffix, 'application/octet-stream')
    
    return FileResponse(
        path=str(resolved),
        media_type=media_type,
        filename=resolved.name
    )


# === Project CRUD Endpoints ===

@app.get("/api/projects", response_model=List[ProjectResponse])
async def list_projects(db: Session = Depends(get_db)):
    """List all projects."""
    projects = db.query(Project).all()
    return projects


@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(request: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project."""
    try:
        preset_type = LoraPresetType(request.lora_preset_type)
    except ValueError:
        preset_type = LoraPresetType.SDXL
    
    project = Project(name=request.name, lora_preset_type=preset_type)
    db.add(project)
    db.commit()
    db.refresh(project)
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        lora_preset_type=project.lora_preset_type.value,
        character_count=0
    )


@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get a project by ID."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        lora_preset_type=project.lora_preset_type.value if project.lora_preset_type else "SDXL",
        character_count=len(project.characters)
    )


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete a project and all its characters."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()
    return {"message": "Project deleted", "id": project_id}


# === Character CRUD Endpoints ===

@app.get("/api/projects/{project_id}/characters", response_model=List[CharacterResponse])
async def list_characters(project_id: int, db: Session = Depends(get_db)):
    """List all characters in a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return [
        CharacterResponse(
            id=c.id,
            project_id=c.project_id,
            name=c.name,
            gender=c.gender.value if c.gender else "neutral",
            reference_count=len(c.reference_images),
            image_count=len(c.dataset_images),
            reference_images_path=c.reference_images_path,
            dataset_images_path=c.dataset_images_path
        )
        for c in project.characters
    ]


@app.post("/api/projects/{project_id}/characters", response_model=CharacterResponse)
async def create_character(
    project_id: int, 
    request: CharacterCreate, 
    db: Session = Depends(get_db)
):
    """Create a new character in a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        gender = Gender(request.gender)
    except ValueError:
        gender = Gender.NEUTRAL
    
    character = Character(
        project_id=project_id,
        name=request.name,
        gender=gender
    )
    db.add(character)
    db.commit()
    db.refresh(character)
    
    return CharacterResponse(
        id=character.id,
        project_id=character.project_id,
        name=character.name,
        gender=character.gender.value,
        reference_count=0,
        image_count=0,
        reference_images_path=None,
        dataset_images_path=None
    )


@app.get("/api/characters/{character_id}", response_model=CharacterResponse)
async def get_character(character_id: int, db: Session = Depends(get_db)):
    """Get a character by ID."""
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    return CharacterResponse(
        id=character.id,
        project_id=character.project_id,
        name=character.name,
        gender=character.gender.value if character.gender else "neutral",
        reference_count=len(character.reference_images),
        image_count=len(character.dataset_images),
        reference_images_path=character.reference_images_path,
        dataset_images_path=character.dataset_images_path
    )


@app.delete("/api/characters/{character_id}")
async def delete_character(character_id: int, db: Session = Depends(get_db)):
    """Delete a character."""
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    db.delete(character)
    db.commit()
    return {"message": "Character deleted", "id": character_id}


# === Reference Images Endpoints ===

@app.post("/api/characters/{character_id}/references")
async def set_reference_images(
    character_id: int,
    request: ReferenceImageCreate,
    db: Session = Depends(get_db)
):
    """Set reference images for a character (replaces existing)."""
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Validate all paths
    for view_type, path in request.images.items():
        is_valid, error = validate_image_path(path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid path for {view_type}: {error}")
    
    # Save reference path if provided
    if request.source_path:
        character.reference_images_path = request.source_path
        db.add(character)
    
    # Run analysis to get embeddings and metrics
    from vision_engine import validate_references
    import json
    
    try:
        # Determine gender string for model selection
        gender_str = character.gender.value if character.gender else "neutral"
        analysis_result = validate_references(request.images, gender=gender_str)
    except Exception as e:
        logger.error(f"Reference analysis failed during set: {e}")
        # We proceed even if analysis fails, but log it
        # Ideally we should warn, but for now we just save paths
        analysis_result = None

    # Clear existing references
    db.query(ReferenceImage).filter(ReferenceImage.character_id == character_id).delete()
    
    # Add new references
    for view_type, path in request.images.items():
        ref = ReferenceImage(
            character_id=character_id,
            path=path,
            view_type=view_type
        )
        
        if analysis_result:
            # Store face embedding
            if view_type in analysis_result.face_embeddings:
                ref.embedding_blob = analysis_result.face_embeddings[view_type].tobytes()
            
            # Store face pose (NEW!)
            if view_type in analysis_result.face_poses:
                import json
                ref.pose_json = json.dumps(analysis_result.face_poses[view_type])
            
            # Store body metrics
            if analysis_result.body_metrics and view_type in analysis_result.body_metrics:
                metrics = analysis_result.body_metrics[view_type]
                
                # Store SMPL-X betas in separate blob column
                if metrics.get('betas') is not None:
                    import numpy as np
                    betas = metrics['betas']
                    if not isinstance(betas, np.ndarray):
                        betas = np.array(betas)
                    ref.betas_blob = betas.astype(np.float32).tobytes()
                
                # Store volume in separate column
                if metrics.get('volume'):
                    ref.volume_estimate = float(metrics['volume'])
                
                # Store 2D ratios in JSON
                if metrics.get('ratios'):
                    import json
                    ref.body_metrics_json = json.dumps({'ratios': metrics['ratios']})
        
        db.add(ref)
    
    db.commit()
    return {"message": "Reference images set and analyzed", "count": len(request.images)}


@app.get("/api/characters/{character_id}/references")
async def get_reference_images(character_id: int, db: Session = Depends(get_db)):
    """Get reference images for a character."""
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    return [
        {
            "id": ref.id,
            "view_type": ref.view_type,
            "path": ref.path
        }
        for ref in character.reference_images
    ]


# === Dataset Images Endpoints ===

@app.get("/api/characters/{character_id}/images", response_model=List[DatasetImageResponse])
async def list_dataset_images(character_id: int, db: Session = Depends(get_db)):
    """List all dataset images for a character."""
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    result = []
    for img in character.dataset_images:
        metrics = img.metrics
        closest_ref_path = None
        if metrics and metrics.closest_face_ref:
            closest_ref_path = metrics.closest_face_ref.path

        result.append(DatasetImageResponse(
            id=img.id,
            original_path=img.original_path,
            filename=Path(img.original_path).name,
            status=img.status.value if img.status else "pending",
            face_similarity=metrics.face_similarity_score if metrics else None,
            body_consistency=metrics.body_consistency_score if metrics else None,
            shot_type=metrics.shot_type if metrics else None,
            limb_ratios=json.loads(metrics.limb_ratios_json) if metrics and metrics.limb_ratios_json else None,
            keypoints=json.loads(metrics.keypoints_json) if metrics and metrics.keypoints_json else None,
            face_bbox=json.loads(metrics.face_bbox_json) if metrics and metrics.face_bbox_json else None,
            closest_face_ref=closest_ref_path
        ))
    
    return result


# === Folder Scanning ===

@app.post("/api/scan-folder", response_model=FolderScanResponse)
async def scan_folder(request: FolderScanRequest, db: Session = Depends(get_db)):
    """Scan a folder for images and create dataset entries."""
    from batch_processor import scan_folder_for_images, create_dataset_entries
    
    character = db.query(Character).filter(Character.id == request.character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    try:
        images = scan_folder_for_images(request.folder_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    new_count = create_dataset_entries(db, request.character_id, images)
    
    # Update character dataset path
    character.dataset_images_path = request.folder_path
    db.add(character)
    db.commit()
    
    return FolderScanResponse(
        folder_path=request.folder_path,
        total_found=len(images),
        new_entries=new_count,
        already_exists=len(images) - new_count
    )


@app.get("/api/list-images")
async def list_images(folder_path: str = Query(..., description="Absolute path to folder")):
    """List images in a folder without creating database entries."""
    from batch_processor import scan_folder_for_images
    
    try:
        images = scan_folder_for_images(folder_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "folder_path": folder_path,
        "images": images
    }


# === Batch Processing ===

@app.post("/api/process/batch", response_model=BatchProcessResponse)
async def start_batch_process(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start batch processing for a character's dataset images."""
    from batch_processor import process_batch
    
    character = db.query(Character).filter(Character.id == request.character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Count images to process
    query = db.query(DatasetImage).filter(DatasetImage.character_id == request.character_id)
    
    if not request.reprocess_all:
        query = query.filter(DatasetImage.status == ImageStatus.PENDING)
        
    pending_count = query.count()
    
    if pending_count == 0:
        msg = "No images found" if request.reprocess_all else "No pending images to process"
        raise HTTPException(status_code=400, detail=msg)
    
    # Create job
    job_id = str(uuid.uuid4())
    job = ProcessingJob(
        id=job_id,
        character_id=request.character_id,
        status=JobStatus.PENDING,
        total_images=pending_count
    )
    db.add(job)
    db.commit()
    
    # Start background processing
    background_tasks.add_task(process_batch, request.character_id, job_id, request.reprocess_all)
    
    return BatchProcessResponse(
        job_id=job_id,
        character_id=request.character_id,
        total_images=pending_count,
        status="pending"
    )


@app.get("/api/process/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get the status of a processing job."""
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value if job.status else "unknown",
        total_images=job.total_images,
        processed_count=job.processed_count,
        error_message=job.error_message
    )


@app.post("/api/process/{job_id}/cancel")
async def cancel_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a running processing job."""
    from batch_processor import request_cancellation
    
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    success = request_cancellation(job_id)
    if success:
        return {"message": "Cancellation requested", "job_id": job_id}
    else:
        raise HTTPException(status_code=400, detail="Job not found or already completed")


# === WebSocket for Progress ===

@app.websocket("/ws/process/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time processing progress."""
    from batch_processor import register_websocket, unregister_websocket
    
    await websocket.accept()
    register_websocket(job_id, websocket)
    
    try:
        while True:
            # Keep connection alive, wait for messages
            data = await websocket.receive_text()
            # Echo back for ping/pong
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        unregister_websocket(job_id, websocket)


# === Thumbnail Service ===

@app.get("/api/image/thumbnail")
async def serve_thumbnail(
    path: str = Query(..., description="Absolute path to image file"),
    size: int = Query(256, description="Maximum thumbnail dimension")
):
    """Serve a cached thumbnail for an image."""
    from thumbnail_service import generate_thumbnail
    
    # Basic path validation (less strict than full images)
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    thumbnail_path = generate_thumbnail(path, size)
    if not thumbnail_path:
        raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
    
    return FileResponse(
        path=thumbnail_path,
        media_type="image/jpeg",
        filename=f"thumb_{resolved.stem}.jpg"
    )

