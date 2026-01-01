"""
Database models and configuration for Character Consistency Validator.
Uses SQLAlchemy ORM with SQLite backend.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey, Enum, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
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
    DENSE = "Dense"


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

    project = relationship("Project", back_populates="characters")
    reference_images = relationship("ReferenceImage", back_populates="character", cascade="all, delete-orphan")
    dataset_images = relationship("DatasetImage", back_populates="character", cascade="all, delete-orphan")


class ReferenceImage(Base):
    """Reference/source-of-truth images for a character."""
    __tablename__ = "reference_images"

    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False)
    path = Column(String(1024), nullable=False)  # Absolute local path
    view_type = Column(String(50))  # e.g., "front", "side", "3/4"
    embedding_blob = Column(LargeBinary)  # Face embedding storage
    smpl_params_blob = Column(LargeBinary)  # Body params storage

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


class ImageMetrics(Base):
    """Analysis metrics for a dataset image."""
    __tablename__ = "image_metrics"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("dataset_images.id"), nullable=False, unique=True)
    face_similarity_score = Column(Float)
    body_consistency_score = Column(Float)
    limb_ratios_json = Column(Text)  # JSON string of limb ratio data
    shot_type = Column(String(50))  # e.g., "close-up", "medium", "full-body"

    image = relationship("DatasetImage", back_populates="metrics")


class Caption(Base):
    """Generated captions for dataset images."""
    __tablename__ = "captions"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("dataset_images.id"), nullable=False)
    model_type = Column(Enum(CaptionModelType), nullable=False)
    text_content = Column(Text, nullable=False)

    image = relationship("DatasetImage", back_populates="captions")


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
