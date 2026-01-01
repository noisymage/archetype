"""
Vision Engine for Character Consistency Validator.

Provides face analysis (InsightFace), pose estimation (YOLO-Pose),
and body shape analysis (SMPLer-X) with graceful degradation.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg'}

# Model paths
MODELS_DIR = Path(__file__).parent / "models"
INSIGHTFACE_DIR = MODELS_DIR / "insightface"
SMPLX_BODY_MODELS_DIR = MODELS_DIR / "smplx" / "body_models"
SMPLX_PRETRAINED_DIR = MODELS_DIR / "smplx" / "pretrained"


def is_supported_image(path: str) -> bool:
    """Check if the file has a supported image format."""
    return Path(path).suffix.lower() in SUPPORTED_FORMATS


@dataclass
class FaceResult:
    """Result from face analysis."""
    detected: bool = False
    embedding: Optional[np.ndarray] = None
    bbox: Optional[list[float]] = None  # [x, y, w, h]
    landmarks: Optional[np.ndarray] = None
    confidence: float = 0.0
    error: Optional[str] = None


@dataclass
class PoseResult:
    """Result from pose estimation."""
    detected: bool = False
    keypoints: Optional[dict[str, dict]] = None  # name -> {x, y, confidence}
    bbox: Optional[list[float]] = None
    confidence: float = 0.0
    error: Optional[str] = None


@dataclass 
class BodyResult:
    """Result from body shape analysis."""
    analyzed: bool = False
    degraded_mode: bool = True  # True if using 2D fallback
    betas: Optional[np.ndarray] = None  # SMPL-X shape parameters
    volume_estimate: Optional[float] = None
    ratios: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ValidationWarning:
    """Warning from reference validation."""
    code: str
    message: str
    severity: str = "warning"  # "warning" or "error"


@dataclass
class ValidationResult:
    """Result from reference image validation."""
    success: bool = False
    degraded_mode: bool = True
    warnings: list[ValidationWarning] = field(default_factory=list)
    master_embedding: Optional[np.ndarray] = None
    body_metrics: Optional[dict] = None
    error: Optional[str] = None


class ModelManager:
    """
    Singleton manager for vision model lifecycle.
    
    Handles loading/unloading of models to manage VRAM.
    Per framework.md: Vision models and LLM must never be loaded simultaneously.
    """
    _instance: Optional['ModelManager'] = None
    
    def __new__(cls) -> 'ModelManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Model instances
        self._face_app = None
        self._pose_model = None
        self._smplx_model = None
        
        # State tracking
        self._face_loaded = False
        self._pose_loaded = False
        self._smplx_loaded = False
        self._smplx_available = False
        self._current_gender = None
        
        # Device detection
        self._device = self._detect_device()
        logger.info(f"ModelManager initialized. Device: {self._device}")
    
    def _detect_device(self) -> str:
        """Detect available compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    @property
    def device(self) -> str:
        """Current compute device."""
        return self._device
    
    @property
    def smplx_available(self) -> bool:
        """Whether SMPLer-X is available (not degraded)."""
        return self._smplx_available
    
    def load_face_model(self) -> bool:
        """
        Load InsightFace model for face detection and embedding.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        if self._face_loaded:
            return True
            
        try:
            from insightface.app import FaceAnalysis
            
            # Create models directory if needed
            INSIGHTFACE_DIR.mkdir(parents=True, exist_ok=True)
            
            # Initialize FaceAnalysis
            self._face_app = FaceAnalysis(
                name="buffalo_l",
                root=str(INSIGHTFACE_DIR),
                providers=self._get_onnx_providers()
            )
            self._face_app.prepare(ctx_id=0 if self._device == "cuda" else -1)
            
            self._face_loaded = True
            logger.info("InsightFace model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            self._face_app = None
            return False
    
    def load_pose_model(self) -> bool:
        """
        Load YOLO-Pose model for 2D pose estimation.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        if self._pose_loaded:
            return True
            
        try:
            from ultralytics import YOLO
            
            # Create models directory if needed
            yolo_dir = MODELS_DIR / "yolo"
            yolo_dir.mkdir(parents=True, exist_ok=True)
            
            # Model path in gitignored directory
            model_path = yolo_dir / "yolov8m-pose.pt"
            
            # Load YOLOv8 pose model (auto-downloads if not present)
            self._pose_model = YOLO(str(model_path) if model_path.exists() else 'yolov8m-pose.pt')
            
            # Move model to correct location if downloaded to CWD
            cwd_model = Path('yolov8m-pose.pt')
            if cwd_model.exists() and not model_path.exists():
                import shutil
                shutil.move(str(cwd_model), str(model_path))
                logger.info(f"Moved YOLO model to {model_path}")
            
            self._pose_loaded = True
            logger.info("YOLO-Pose model loaded successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"Ultralytics not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize YOLO-Pose: {e}")
            return False

    
    def load_mesh_model(self, gender: str = "neutral") -> bool:
        """
        Load SMPLer-X model for 3D mesh recovery.
        
        Args:
            gender: One of "male", "female", "neutral"
            
        Returns:
            True if loaded successfully, False if degraded to 2D.
        """
        if self._smplx_loaded and self._current_gender == gender:
            return self._smplx_available
        
        # Unload previous if different gender
        if self._smplx_loaded and self._current_gender != gender:
            self._unload_mesh_model()
        
        try:
            import smplx
            import torch
            
            # Model file mapping
            model_files = {
                "male": "SMPLX_MALE.npz", 
                "female": "SMPLX_FEMALE.npz",
                "neutral": "SMPLX_NEUTRAL.npz"
            }
            
            # Try gender-specific, fall back to neutral
            model_file = model_files.get(gender, "SMPLX_NEUTRAL.npz")
            model_path = SMPLX_BODY_MODELS_DIR / model_file
            
            if not model_path.exists():
                # Try neutral as fallback
                if gender != "neutral":
                    logger.warning(f"Gender-specific model {model_file} not found, trying neutral")
                    model_path = SMPLX_BODY_MODELS_DIR / "SMPLX_NEUTRAL.npz"
                
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"SMPL-X body model not found. Please download from smpl-x.is.tue.mpg.de "
                        f"and place in {SMPLX_BODY_MODELS_DIR}"
                    )
            
            # Check for SMPLer-X checkpoint
            checkpoint_path = SMPLX_PRETRAINED_DIR / "smpler_x_h32.pth.tar"
            if not checkpoint_path.exists():
                # Try smaller model
                checkpoint_path = SMPLX_PRETRAINED_DIR / "smpler_x_s32.pth.tar"
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"SMPLer-X checkpoint not found. Please download from HuggingFace "
                        f"and place in {SMPLX_PRETRAINED_DIR}"
                    )
            
            # Load SMPL-X body model
            device = torch.device(self._device if self._device != "mps" else "cpu")
            self._smplx_model = smplx.create(
                str(SMPLX_BODY_MODELS_DIR),
                model_type='smplx',
                gender=gender,
                use_face_contour=False,
                num_betas=10,
                num_expression_coeffs=10
            ).to(device)
            
            self._smplx_loaded = True
            self._smplx_available = True
            self._current_gender = gender
            logger.info(f"SMPLer-X model loaded successfully (gender={gender})")
            return True
            
        except FileNotFoundError as e:
            logger.warning(f"SMPLer-X model files missing: {e}")
            logger.warning("Degrading to 2D keypoint-based body analysis")
            self._smplx_available = False
            self._smplx_loaded = True  # Mark as "loaded" in degraded mode
            self._current_gender = gender
            return False
            
        except ImportError as e:
            logger.warning(f"SMPLer-X dependencies not available: {e}")
            logger.warning("Degrading to 2D keypoint-based body analysis")
            self._smplx_available = False
            self._smplx_loaded = True
            self._current_gender = gender
            return False
            
        except Exception as e:
            logger.error(f"Failed to load SMPLer-X: {e}")
            logger.warning("Degrading to 2D keypoint-based body analysis")
            self._smplx_available = False
            self._smplx_loaded = True
            self._current_gender = gender
            return False
    
    def _unload_mesh_model(self):
        """Unload SMPL-X model."""
        if self._smplx_model is not None:
            del self._smplx_model
            self._smplx_model = None
        self._smplx_loaded = False
        self._smplx_available = False
        self._current_gender = None
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def unload_all(self):
        """
        Unload all models to free VRAM.
        
        Call this before loading LLM models per framework.md requirements.
        """
        logger.info("Unloading all vision models...")
        
        # Unload face model
        if self._face_app is not None:
            del self._face_app
            self._face_app = None
        self._face_loaded = False
        
        # Unload pose model
        if self._pose_model is not None:
            del self._pose_model
            self._pose_model = None
        self._pose_loaded = False
        
        # Unload mesh model
        self._unload_mesh_model()
        
        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("All vision models unloaded")
    
    def _get_onnx_providers(self) -> list[str]:
        """Get ONNX Runtime execution providers based on device."""
        if self._device == "cuda":
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self._device == "mps":
            # CoreML for Apple Silicon
            return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']


class FaceAnalyzer:
    """
    Face detection and embedding extraction using InsightFace.
    """
    
    def __init__(self, model_manager: ModelManager):
        self._manager = model_manager
    
    def analyze(self, image_path: str) -> FaceResult:
        """
        Analyze a face in an image.
        
        Args:
            image_path: Absolute path to image file.
            
        Returns:
            FaceResult with embedding, bbox, landmarks, or error.
        """
        result = FaceResult()
        
        # Validate input
        if not os.path.exists(image_path):
            result.error = f"Image not found: {image_path}"
            return result
            
        if not is_supported_image(image_path):
            result.error = f"Unsupported format. Use: {SUPPORTED_FORMATS}"
            return result
        
        # Ensure model is loaded
        if not self._manager.load_face_model():
            result.error = "Failed to load face model"
            return result
        
        try:
            import cv2
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                result.error = f"Failed to read image: {image_path}"
                return result
            
            # Run face detection
            faces = self._manager._face_app.get(img)
            
            if not faces:
                result.error = "No face detected in image"
                return result
            
            # Use the largest/most confident face
            face = max(faces, key=lambda f: f.det_score)
            
            result.detected = True
            result.embedding = face.embedding
            result.bbox = face.bbox.tolist()  # [x1, y1, x2, y2]
            result.landmarks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else face.kps
            result.confidence = float(face.det_score)
            
            return result
            
        except Exception as e:
            result.error = f"Face analysis failed: {str(e)}"
            logger.exception("Face analysis error")
            return result


class PoseEstimator:
    """
    2D pose estimation using Ultralytics YOLO-Pose.
    """
    
    # COCO keypoint names (17 keypoints)
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_manager: ModelManager):
        self._manager = model_manager
    
    def estimate(self, image_path: str) -> PoseResult:
        """
        Estimate 2D pose keypoints using YOLO-Pose.
        
        Args:
            image_path: Absolute path to image file.
            
        Returns:
            PoseResult with keypoints or error.
        """
        result = PoseResult()
        
        # Validate input
        if not os.path.exists(image_path):
            result.error = f"Image not found: {image_path}"
            return result
            
        if not is_supported_image(image_path):
            result.error = f"Unsupported format. Use: {SUPPORTED_FORMATS}"
            return result
        
        # Ensure model is loaded
        if not self._manager.load_pose_model():
            result.error = "Failed to load pose model"
            return result
        
        try:
            # Run YOLO-Pose inference
            predictions = self._manager._pose_model(image_path, verbose=False)
            
            if not predictions or len(predictions) == 0:
                result.error = "No predictions returned"
                return result
            
            pred = predictions[0]
            
            # Check if any people detected
            if pred.keypoints is None or len(pred.keypoints) == 0:
                result.error = "No person detected in image"
                return result
            
            # Get the most confident person (largest bounding box as proxy)
            if pred.boxes is not None and len(pred.boxes) > 0:
                # Find person with largest bbox area
                boxes = pred.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_idx = int(areas.argmax())
                result.bbox = boxes[best_idx].tolist()
            else:
                best_idx = 0
            
            # Extract keypoints for best person
            keypoints_data = pred.keypoints[best_idx]
            xy = keypoints_data.xy.cpu().numpy()[0]  # Shape: (17, 2)
            conf = keypoints_data.conf.cpu().numpy()[0] if keypoints_data.conf is not None else np.ones(17)
            
            # Build keypoints dict
            keypoints = {}
            for i, name in enumerate(self.KEYPOINT_NAMES):
                keypoints[name] = {
                    'x': float(xy[i, 0]),
                    'y': float(xy[i, 1]),
                    'confidence': float(conf[i])
                }
            
            result.detected = True
            result.keypoints = keypoints
            result.confidence = float(np.mean(conf))
            
            return result
            
        except Exception as e:
            result.error = f"Pose estimation failed: {str(e)}"
            logger.exception("Pose estimation error")
            return result


class BodyAnalyzer:
    """
    Body shape and proportion analysis.
    
    Uses SMPLer-X for 3D mesh recovery when available,
    falls back to 2D keypoint-based metrics.
    """
    
    def __init__(self, model_manager: ModelManager):
        self._manager = model_manager
    
    def analyze(self, image_path: str, keypoints: Optional[dict] = None, 
                gender: str = "neutral") -> BodyResult:
        """
        Analyze body shape and proportions.
        
        Args:
            image_path: Absolute path to image file.
            keypoints: Pre-computed keypoints (optional).
            gender: Character gender for model selection.
            
        Returns:
            BodyResult with betas, volume, ratios, or error.
        """
        result = BodyResult()
        
        # Validate input
        if not os.path.exists(image_path):
            result.error = f"Image not found: {image_path}"
            return result
        
        # Attempt to load mesh model
        smplx_loaded = self._manager.load_mesh_model(gender)
        result.degraded_mode = not smplx_loaded
        
        if smplx_loaded and self._manager._smplx_model is not None:
            # Full SMPLer-X analysis
            return self._analyze_with_smplx(image_path, result)
        else:
            # Degraded: 2D keypoint-based analysis
            return self._analyze_with_keypoints(image_path, keypoints, result)
    
    def _analyze_with_smplx(self, image_path: str, result: BodyResult) -> BodyResult:
        """Full 3D analysis using SMPLer-X."""
        try:
            # TODO: Implement full SMPLer-X inference pipeline
            # This requires:
            # 1. Image preprocessing
            # 2. Running SMPLer-X model
            # 3. Extracting betas (shape parameters)
            # 4. Computing volume from mesh
            
            result.error = "SMPLer-X inference not yet fully implemented"
            result.analyzed = False
            return result
            
        except Exception as e:
            result.error = f"SMPLer-X analysis failed: {str(e)}"
            logger.exception("SMPLer-X analysis error")
            return result
    
    def _analyze_with_keypoints(self, image_path: str, 
                                 keypoints: Optional[dict],
                                 result: BodyResult) -> BodyResult:
        """Fallback 2D analysis using keypoints."""
        try:
            if keypoints is None:
                result.error = "No keypoints available for 2D analysis"
                return result
            
            # Compute body ratios from keypoints
            ratios = self._compute_ratios(keypoints)
            
            result.analyzed = True
            result.ratios = ratios
            result.degraded_mode = True
            
            return result
            
        except Exception as e:
            result.error = f"Keypoint analysis failed: {str(e)}"
            logger.exception("Keypoint analysis error")
            return result
    
    def _compute_ratios(self, keypoints: dict) -> dict:
        """Compute skeletal ratios from 2D keypoints."""
        ratios = {}
        
        def distance(p1: str, p2: str) -> Optional[float]:
            if p1 not in keypoints or p2 not in keypoints:
                return None
            k1, k2 = keypoints[p1], keypoints[p2]
            if k1['confidence'] < 0.3 or k2['confidence'] < 0.3:
                return None
            return np.sqrt((k1['x'] - k2['x'])**2 + (k1['y'] - k2['y'])**2)
        
        # Shoulder width
        shoulder_width = distance('left_shoulder', 'right_shoulder')
        hip_width = distance('left_hip', 'right_hip')
        
        if shoulder_width and hip_width:
            ratios['shoulder_to_hip'] = shoulder_width / hip_width
        
        # Arm proportions
        upper_arm_l = distance('left_shoulder', 'left_elbow')
        lower_arm_l = distance('left_elbow', 'left_wrist')
        if upper_arm_l and lower_arm_l:
            ratios['upper_to_lower_arm'] = upper_arm_l / lower_arm_l
        
        # Leg proportions
        upper_leg_l = distance('left_hip', 'left_knee')
        lower_leg_l = distance('left_knee', 'left_ankle')
        if upper_leg_l and lower_leg_l:
            ratios['upper_to_lower_leg'] = upper_leg_l / lower_leg_l
        
        # Torso to leg ratio
        torso_height = distance('left_shoulder', 'left_hip')
        leg_height = distance('left_hip', 'left_ankle')
        if torso_height and leg_height:
            ratios['torso_to_leg'] = torso_height / leg_height
        
        return ratios


def validate_references(images: dict[str, str], gender: str = "neutral") -> ValidationResult:
    """
    Validate a set of reference images for consistency.
    
    Args:
        images: Dict mapping view type to image path:
            - head_front, head_45l, head_45r (Face Analysis only)
            - body_front, body_side, body_posterior (Full analysis)
        gender: Character gender for SMPL-X model selection.
        
    Returns:
        ValidationResult with success/fail, warnings, and extracted data.
    """
    result = ValidationResult()
    manager = ModelManager()
    
    face_analyzer = FaceAnalyzer(manager)
    pose_estimator = PoseEstimator(manager)
    body_analyzer = BodyAnalyzer(manager)
    
    # Validate required images
    required_head = {'head_front', 'head_45l', 'head_45r'}
    required_body = {'body_front', 'body_side', 'body_posterior'}
    
    provided = set(images.keys())
    missing_head = required_head - provided
    missing_body = required_body - provided
    
    if missing_head:
        result.warnings.append(ValidationWarning(
            code="MISSING_HEAD_REFS",
            message=f"Missing head reference images: {missing_head}",
            severity="error"
        ))
    
    if missing_body:
        result.warnings.append(ValidationWarning(
            code="MISSING_BODY_REFS", 
            message=f"Missing body reference images: {missing_body}",
            severity="error"
        ))
    
    if missing_head or missing_body:
        result.error = "Required reference images missing"
        return result
    
    # === Head Group Analysis ===
    head_embeddings = []
    for view in ['head_front', 'head_45l', 'head_45r']:
        face_result = face_analyzer.analyze(images[view])
        if face_result.detected and face_result.embedding is not None:
            head_embeddings.append(face_result.embedding)
        else:
            result.warnings.append(ValidationWarning(
                code="HEAD_FACE_NOT_DETECTED",
                message=f"No face detected in {view}: {face_result.error}",
                severity="warning"
            ))
    
    # Compute Master Identity Vector
    if head_embeddings:
        result.master_embedding = np.mean(head_embeddings, axis=0)
    else:
        result.warnings.append(ValidationWarning(
            code="NO_HEAD_EMBEDDINGS",
            message="Could not establish Master Identity Vector - no faces detected",
            severity="error"
        ))
    
    # === Body Group Analysis ===
    body_views = ['body_front', 'body_side', 'body_posterior']
    body_metrics = {}
    pose_data = {}
    
    for view in body_views:
        # Pose estimation
        pose_result = pose_estimator.estimate(images[view])
        if pose_result.detected:
            pose_data[view] = pose_result.keypoints
        
        # Body analysis
        body_result = body_analyzer.analyze(
            images[view], 
            keypoints=pose_result.keypoints if pose_result.detected else None,
            gender=gender
        )
        if body_result.analyzed:
            body_metrics[view] = {
                'degraded_mode': body_result.degraded_mode,
                'betas': body_result.betas.tolist() if body_result.betas is not None else None,
                'volume': body_result.volume_estimate,
                'ratios': body_result.ratios
            }
        
        result.degraded_mode = result.degraded_mode or body_result.degraded_mode
    
    # === Cross-Check: Face identity in body images ===
    if result.master_embedding is not None:
        for view in ['body_front', 'body_side']:
            face_result = face_analyzer.analyze(images[view])
            if face_result.detected and face_result.embedding is not None:
                # Cosine similarity
                similarity = np.dot(result.master_embedding, face_result.embedding) / (
                    np.linalg.norm(result.master_embedding) * np.linalg.norm(face_result.embedding)
                )
                if similarity < 0.75:
                    result.warnings.append(ValidationWarning(
                        code="IDENTITY_MISMATCH",
                        message=f"Face in {view} (sim={similarity:.2f}) may not match head references",
                        severity="warning"
                    ))
    
    # === Pose & Alignment Checks ===
    if len(pose_data) >= 2:
        alignment_warnings = _check_pose_alignment(pose_data)
        result.warnings.extend(alignment_warnings)
    
    # === Volume Consistency Check ===
    if len(body_metrics) >= 2:
        volume_warnings = _check_volume_consistency(body_metrics)
        result.warnings.extend(volume_warnings)
    
    # === Ratio Consistency Check ===
    if len(body_metrics) >= 2:
        ratio_warnings = _check_ratio_consistency(body_metrics)
        result.warnings.extend(ratio_warnings)
    
    # Final result
    result.body_metrics = body_metrics
    result.success = not any(w.severity == "error" for w in result.warnings)
    
    return result


def _check_pose_alignment(pose_data: dict) -> list[ValidationWarning]:
    """Check if body views are at the same scale."""
    warnings = []
    
    # Compare normalized Y-coordinates of key joints
    # TODO: Implement scale alignment check
    
    return warnings


def _check_volume_consistency(body_metrics: dict) -> list[ValidationWarning]:
    """Check if volume estimates are consistent across views."""
    warnings = []
    
    volumes = [m.get('volume') for m in body_metrics.values() if m.get('volume')]
    if len(volumes) >= 2:
        mean_vol = np.mean(volumes)
        max_var = max(abs(v - mean_vol) / mean_vol for v in volumes)
        if max_var > 0.10:  # 10% variance threshold
            warnings.append(ValidationWarning(
                code="VOLUME_INCONSISTENCY",
                message=f"Volume estimates vary by {max_var*100:.1f}% across views",
                severity="warning"
            ))
    
    return warnings


def _check_ratio_consistency(body_metrics: dict) -> list[ValidationWarning]:
    """Check if skeletal ratios are consistent across views."""
    warnings = []
    
    # Collect all ratio keys
    all_ratios = {}
    for view, metrics in body_metrics.items():
        for key, value in metrics.get('ratios', {}).items():
            if key not in all_ratios:
                all_ratios[key] = []
            all_ratios[key].append(value)
    
    # Check variance for each ratio
    for ratio_name, values in all_ratios.items():
        if len(values) >= 2:
            mean_val = np.mean(values)
            max_var = max(abs(v - mean_val) / mean_val for v in values) if mean_val != 0 else 0
            if max_var > 0.10:  # 10% variance threshold
                warnings.append(ValidationWarning(
                    code="RATIO_INCONSISTENCY",
                    message=f"Ratio '{ratio_name}' varies by {max_var*100:.1f}% across views",
                    severity="warning"
                ))
    
    return warnings


def analyze_single_image(image_path: str) -> dict:
    """
    Analyze a single image for face, pose, and shot type.
    
    Args:
        image_path: Absolute path to image file.
        
    Returns:
        Dict with face_detected, face_bbox, body_detected, keypoints, shot_type.
    """
    manager = ModelManager()
    face_analyzer = FaceAnalyzer(manager)
    pose_estimator = PoseEstimator(manager)
    
    result = {
        'face_detected': False,
        'face_bbox': None,
        'body_detected': False,
        'keypoints': None,
        'shot_type': 'unknown',
        'error': None
    }
    
    # Face analysis
    face_result = face_analyzer.analyze(image_path)
    if face_result.detected:
        result['face_detected'] = True
        result['face_bbox'] = face_result.bbox
    
    # Pose estimation
    pose_result = pose_estimator.estimate(image_path)
    if pose_result.detected:
        result['body_detected'] = True
        result['keypoints'] = pose_result.keypoints
    
    # Determine shot type based on detected elements
    if result['body_detected'] and result['keypoints']:
        # Check which body parts are visible
        kp = result['keypoints']
        has_full_body = all(
            k in kp and kp[k].get('confidence', 0) > 0.3 
            for k in ['left_ankle', 'right_ankle', 'left_shoulder', 'right_shoulder']
        )
        has_upper = all(
            k in kp and kp[k].get('confidence', 0) > 0.3
            for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        )
        
        if has_full_body:
            result['shot_type'] = 'full-body'
        elif has_upper:
            result['shot_type'] = 'medium'
        elif result['face_detected']:
            result['shot_type'] = 'close-up'
    elif result['face_detected']:
        result['shot_type'] = 'close-up'
    
    return result
