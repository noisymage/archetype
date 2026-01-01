"""
FastAPI application entry point for Character Consistency Validator.
"""
import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from database import init_db

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


# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - initialize database on startup."""
    init_db()
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
            body_metrics=result.body_metrics,
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
