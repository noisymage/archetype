"""
Database models and configuration for Character Consistency Validator.
Uses SQLAlchemy ORM with SQLite backend.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey, Enum, LargeBinary, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum
import os

# Database path - stored in backend directory
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "archetype.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class LoraPresetType(enum.Enum):
    SDXL = "SDXL"
    FLUX = "Flux"
    FACE_ONLY = "Face-Only"


class ImageStatus(enum.Enum):
    PENDING = "pending"
    ANALYZED = "analyzed"
    REJECTED = "rejected"
    APPROVED = "approved"


class CaptionModelType(enum.Enum):
    SDXL = "SDXL"
    FLUX = "Flux"
    QWEN_IMAGE = "Qwen-Image"
    Z_IMAGE = "Z-Image"


class Gender(enum.Enum):
    """Character gender for SMPL-X model selection."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class JobStatus(enum.Enum):
    """Status of a batch processing job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Project(Base):
    """Project container for organizing characters and datasets."""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    lora_preset_type = Column(Enum(LoraPresetType), default=LoraPresetType.SDXL)

    characters = relationship("Character", back_populates="project", cascade="all, delete-orphan")


class Character(Base):
    """Character entity within a project."""
    __tablename__ = "characters"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    gender = Column(Enum(Gender), default=Gender.NEUTRAL)

    project = relationship("Project", back_populates="characters")
    reference_images = relationship("ReferenceImage", back_populates="character", cascade="all, delete-orphan")
    dataset_images = relationship("DatasetImage", back_populates="character", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="character", cascade="all, delete-orphan")

    # New fields for path persistence
    reference_images_path = Column(String(1024), nullable=True)
    dataset_images_path = Column(String(1024), nullable=True)

    @property
    def reference_count(self):
        return len(self.reference_images)

    @property
    def image_count(self):
        return len(self.dataset_images)

def migrate_db():
    """Simple migration to add new columns if they don't exist."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    
    # Migrate characters table
    columns = [c['name'] for c in inspector.get_columns('characters')]
    
    with engine.connect() as conn:
        if 'reference_images_path' not in columns:
            conn.execute(text("ALTER TABLE characters ADD COLUMN reference_images_path VARCHAR(1024)"))
            print("Added reference_images_path column")
        
        if 'dataset_images_path' not in columns:
            conn.execute(text("ALTER TABLE characters ADD COLUMN dataset_images_path VARCHAR(1024)"))
            print("Added dataset_images_path column")
        
        conn.commit()
    
    # Migrate reference_images table
    ref_columns = [c['name'] for c in inspector.get_columns('reference_images')]
    
    with engine.connect() as conn:
        if 'betas_blob' not in ref_columns:
            conn.execute(text("ALTER TABLE reference_images ADD COLUMN betas_blob BLOB"))
            print("Added betas_blob column to reference_images")
        
        if 'volume_estimate' not in ref_columns:
            conn.execute(text("ALTER TABLE reference_images ADD COLUMN volume_estimate FLOAT"))
            print("Added volume_estimate column to reference_images")
        
        if 'adaface_embedding_blob' not in ref_columns:
            conn.execute(text("ALTER TABLE reference_images ADD COLUMN adaface_embedding_blob BLOB"))
            print("Added adaface_embedding_blob column to reference_images")
        
        conn.commit()
    
    # Migrate image_metrics table
    try:
        metrics_columns = [c['name'] for c in inspector.get_columns('image_metrics')]
        
        with engine.connect() as conn:
            if 'face_model_used' not in metrics_columns:
                conn.execute(text("ALTER TABLE image_metrics ADD COLUMN face_model_used VARCHAR(20)"))
                print("Added face_model_used column to image_metrics")
            
            conn.commit()
    except Exception:
        pass  # Table may not exist yet
    
    # Create image_descriptions table if it doesn't exist
    try:
        if 'image_descriptions' not in inspector.get_table_names():
            ImageDescription.__table__.create(engine)
            print("Created image_descriptions table")
    except Exception as e:
        print(f"Note: image_descriptions table handling: {e}")



class ReferenceImage(Base):
    """Reference/source-of-truth images for a character."""
    __tablename__ = "reference_images"

    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False)
    path = Column(String(500), nullable=False)
    view_type = Column(String(50))  # e.g., "head_front", "body_front"
    embedding_blob = Column(LargeBinary)  # InsightFace embedding as binary
    adaface_embedding_blob = Column(LargeBinary)  # AdaFace embedding as binary
    pose_json = Column(Text)  # Head pose: {"yaw": float, "pitch": float, "roll": float}
    
    # Body metrics for body references
    betas_blob = Column(LargeBinary)  # SMPL-X beta parameters (shape) as binary
    volume_estimate = Column(Float)  # 3D volume proxy from mesh
    body_metrics_json = Column(Text)  # 2D skeletal ratios and other body data

    character = relationship("Character", back_populates="reference_images")


class DatasetImage(Base):
    """Dataset images to be validated against references."""
    __tablename__ = "dataset_images"

    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False)
    original_path = Column(String(1024), nullable=False)  # Absolute local path
    status = Column(Enum(ImageStatus), default=ImageStatus.PENDING)

    character = relationship("Character", back_populates="dataset_images")
    metrics = relationship("ImageMetrics", back_populates="image", uselist=False, cascade="all, delete-orphan")
    captions = relationship("Caption", back_populates="image", cascade="all, delete-orphan")
    description = relationship("ImageDescription", back_populates="image", uselist=False, cascade="all, delete-orphan")


class ImageMetrics(Base):
    """Analysis metrics for a dataset image."""
    __tablename__ = "image_metrics"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("dataset_images.id"), nullable=False, unique=True)
    face_similarity_score = Column(Float)
    face_model_used = Column(String(20))  # "insightface", "adaface", or None
    body_consistency_score = Column(Float)
    limb_ratios_json = Column(Text)  # JSON string of limb ratio data
    shot_type = Column(String(50))  # e.g., "close-up", "medium", "full-body"
    keypoints_json = Column(Text)  # JSON string of YOLO keypoints for overlay
    face_bbox_json = Column(Text)  # JSON string of face bounding box [x1, y1, x2, y2]
    face_pose_json = Column(Text)  # JSON string of head pose {"yaw": ..., "pitch": ..., "roll": ...}
    closest_face_ref_id = Column(Integer, ForeignKey("reference_images.id"), nullable=True)

    image = relationship("DatasetImage", back_populates="metrics")
    closest_face_ref = relationship("ReferenceImage")


class Caption(Base):
    """Generated captions for dataset images."""
    __tablename__ = "captions"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("dataset_images.id"), nullable=False)
    model_type = Column(Enum(CaptionModelType), nullable=False)
    text_content = Column(Text, nullable=False)

    image = relationship("DatasetImage", back_populates="captions")


class ImageDescription(Base):
    """Rich textual description of a dataset image from vision LLM."""
    __tablename__ = "image_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("dataset_images.id"), nullable=False, unique=True)
    
    # Structured metadata
    shot_type = Column(String(50))              # close-up, medium, full-body
    pose_description = Column(Text)             # e.g., "standing with arms crossed"
    expression = Column(String(100))            # e.g., "confident smile"
    clothing_description = Column(Text)         # e.g., "blue dress with white patterns"
    lighting_description = Column(String(255))  # e.g., "dramatic side lighting"
    background_description = Column(Text)       # e.g., "urban street at night"
    
    # Full analysis
    full_description = Column(Text)             # Detailed paragraph description
    
    # LLM assessment
    quality_notes = Column(Text)                # Any issues the LLM flagged
    
    # Generation metadata
    llm_provider = Column(String(50))           # "gemini", "ollama"
    llm_model = Column(String(100))             # "gemini-2.0-flash", "llava:13b"
    generated_at = Column(DateTime)
    
    image = relationship("DatasetImage", back_populates="description")


class ProcessingJob(Base):
    """Track batch processing jobs for characters."""
    __tablename__ = "processing_jobs"

    id = Column(String(36), primary_key=True)  # UUID
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    total_images = Column(Integer, default=0)
    processed_count = Column(Integer, default=0)
    cancelled = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    character = relationship("Character", back_populates="processing_jobs")


def get_db():
    """Dependency for FastAPI to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
