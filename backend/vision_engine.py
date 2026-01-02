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
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import sys
import cv2
import torch
from torchvision import transforms

# --- SMPLest-X Path Setup ---
BASE_DIR = Path(__file__).parent.parent 
# We need to target the submodule root explicitly for its internal imports (e.g. 'import common.utils')
# SMPL-X functionality removed - was non-deterministic and unsuitable for consistency matching
# See optimization_based_smpl_fitting.md for future deterministic implementation
SMPLEST_X_AVAILABLE = False


# Configure logging
logger = logging.getLogger(__name__)

try:
    # Try importing essential SMPLest-X modules
    # We do this after path injection
    # SMPLest-X is no longer used directly, so these imports are removed.
    pass
except ImportError as e:
    logger.warning(f"SMPLest-X imports failed (expected as it's disabled): {e}")
    # We allow it to fail here, ModelManager will report it later
    pass


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
    pose: Optional[dict] = None  # {"yaw": float, "pitch": float, "roll": float}
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
    face_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    face_poses: dict[str, dict] = field(default_factory=dict)  # view_type -> {"yaw": ..., "pitch": ..., "roll": ...}
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
        # SMPLest-X removed
        
        # State tracking
        self._face_loaded = False
        self._pose_loaded = False
        self._smplx_loaded = False # Still track if an attempt was made
        
        # Device detection
        import torch
        self._device = "cpu"
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device = "mps"
        
        logger.info(f"Vision Engine initialized on {self.device}")
    
    @property
    def device(self) -> str:
        """Current compute device."""
        return self._device
    
    @property
    def smplx_available(self) -> bool:
        """Whether SMPLer-X is available (not degraded)."""
        # SMPL-X is no longer available directly
        return False
    
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
            self._face_app.prepare(ctx_id=0 if self.device == "cuda" else -1)
            
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

    
            self._smplx_wrapper = None
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


def estimate_head_pose(landmarks: np.ndarray, image_shape: tuple) -> Optional[dict]:
    """
    Estimate head pose (yaw, pitch, roll) from facial landmarks using PnP algorithm.
    
    Args:
        landmarks: Facial landmarks array (5 or 106 points)
        image_shape: (height, width) of the image
    
    Returns:
        {"yaw": float, "pitch": float, "roll": float} in degrees, or None if failed
    """
    try:
        import cv2
        
        # InsightFace returns 106 or 5 landmarks depending on model
        # We need at least 6 points for reliable pose estimation
        if landmarks.shape[0] < 6:
            logger.debug(f"Not enough landmarks for pose estimation: {landmarks.shape}")
            return None
        
        # Extract 6 key facial points
        if len(landmarks.shape) == 2 and landmarks.shape[1] >= 2:
            # Use first 6 points from 106-landmark model
            # These correspond to: various face outline and feature points
            image_points = landmarks[:6, :2].copy().astype(np.float32)
        else:
            logger.debug(f"Unexpected landmark shape: {landmarks.shape}")
            return None
        
        # Ensure it's a contiguous array (OpenCV requirement)
        image_points = np.ascontiguousarray(image_points, dtype=np.float32)
        
        # Verify we have exactly 6 valid 2D points
        if image_points.shape != (6, 2):
            logger.debug(f"Invalid image_points shape after processing: {image_points.shape}")
            return None
        
        # 3D model points of a generic face (in mm)
        # Approximate positions for first 6 landmarks from InsightFace 106-point model
        model_points = np.array([
            (0.0, -330.0, -65.0),      # Chin center
            (-225.0, 170.0, -135.0),   # Left eye outer corner
            (225.0, 170.0, -135.0),    # Right eye outer corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0),   # Right mouth corner
            (0.0, 0.0, 0.0),           # Nose tip
        ], dtype=np.float32)
        
        # Camera internals (approximate monocular camera)
        h, w = image_shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Solve PnP using EPNP (works with 4+ points, more robust than ITERATIVE)
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP  # Changed from ITERATIVE
        )
        
        if not success:
            return None
        
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Create projection matrix
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        
        # Decompose projection matrix to get Euler angles
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        # Extract pitch, yaw, roll (in degrees)
        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])
        
        # Normalize angles to reasonable ranges
        # Yaw: -180 to 180
        # Pitch: -90 to 90  
        # Roll: -180 to 180
        yaw = ((yaw + 180) % 360) - 180
        pitch = ((pitch + 90) % 180) - 90
        roll = ((roll + 180) % 360) - 180
        
        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll
        }
        
    except Exception as e:
        logger.warning(f"Head pose estimation failed: {e}")
        return None


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
            
            # Calculate head pose from landmarks
            if result.landmarks is not None:
                result.pose = estimate_head_pose(result.landmarks, img.shape)
            
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
                bbox: Optional[list] = None, gender: str = "neutral",
                reference_betas: Optional[list[np.ndarray]] = None,
                reference_ratios: Optional[list[dict]] = None) -> BodyResult:
        """
        Analyze body shape and proportions using both 3D and 2D methods.
        
        Args:
            image_path: Absolute path to image file.
            keypoints: Pre-computed keypoints (optional).
            bbox: Pre-computed bounding box [x1, y1, x2, y2] (optional).
            gender: Character gender for model selection.
            
        Returns:
            BodyResult with both 3D and 2D metrics when available.
        """
        result = BodyResult()
        
        # Validate input
        if not os.path.exists(image_path):
            result.error = f"Image not found: {image_path}"
            return result
        
        # Initialize dual metrics containers
        metrics_3d = None
        preferred = "none"
        
        
        # 3D analysis disabled (SMPL-X removed)
        # Always use 2D skeletal ratios only
        
        # Always attempt 2D analysis if keypoints available
        if keypoints:
            try:
                metrics_2d = self._analyze_with_keypoints(keypoints, reference_ratios)
                if metrics_2d and metrics_2d.get('success'):
                    if preferred == "none":
                        preferred = "2d"
            except Exception as e:
                logger.warning(f"2D analysis failed: {e}")
        
        # Populate result
        result.analyzed = (metrics_3d is not None or metrics_2d is not None)
        result.degraded_mode = (preferred == "2d")
        
        if metrics_3d and metrics_3d.get("success"):
            import numpy as np
            result.volume_estimate = metrics_3d.get("volume")
            result.betas = np.array(metrics_3d.get("betas")) if metrics_3d.get("betas") else None

        # Store dual metrics in ratios field (will be restructured in batch_processor)
        result.ratios = {
            "metrics_3d": metrics_3d,
            "metrics_2d": metrics_2d,
            "preferred": preferred
        }
        
        return result
    
    def _analyze_with_smplx(self, image_path: str, bbox: Optional[list] = None, 
                             reference_betas: Optional[list[np.ndarray]] = None) -> Optional[dict]:
        """
        Full 3D analysis using SMPLest-X.
        
        Args:
            image_path: Path to image
            bbox: Optional [x1, y1, x2, y2] bounding box to skip detection
            
        Returns:
            Dict with 3D metrics or None if failed.
        """
        wrapper = self._manager._smplx_wrapper
        
        if not wrapper or not wrapper.initialized:
            return None
            
        try:
            import cv2
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Use provided bbox or detect new one
            final_bbox = None
            
            if bbox:
                final_bbox = bbox
            else:
                # Fallback to internal detection
                if not self._manager._pose_loaded:
                    self._manager.load_pose_model()
                
                pose_res = self._manager._pose_model(image_path, verbose=False)[0]
                if pose_res.boxes and len(pose_res.boxes) > 0:
                     # Best person
                    boxes = pose_res.boxes.xyxy.cpu().numpy()
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    best_idx = int(areas.argmax())
                    final_bbox = boxes[best_idx].tolist()
            
            if not final_bbox:
                return {"error": "No person detected for 3D Analysis"}

            # Run SMPLest-X inference
            inference_out = wrapper.run_inference(img, final_bbox)
            
            if inference_out:
                # Calculate 3D-based ratios from mesh if available
                ratios = {}
                
                # DISABLED: Beta-based consistency is unreliable due to model non-determinism
                # Same image produces different betas each run, making it unsuitable for identity matching
                # TODO: Replace with optimization-based SMPL fitting for deterministic results
                # See: optimization_based_smpl_fitting.md for implementation guide
                consistency_score = None
                
                return {
                    "success": True,
                    "betas": inference_out.get('betas'),
                    "volume": inference_out.get('volume_proxy'),
                    "ratios": ratios,
                    "consistency_score": consistency_score
                }
            
            return None
            
        except Exception as e:
            logger.exception(f"SMPLest-X analysis error: {e}")
            return None
    
    def _analyze_with_keypoints(self, keypoints: dict, 
                                  reference_ratios: Optional[list[dict]] = None) -> Optional[dict]:
        """
        2D analysis using keypoints.
        
        Returns:
            Dict with 2D metrics or None if failed.
        """
        try:
            if not keypoints:
                return None
            
            # Compute body ratios from keypoints
            ratios = self._compute_ratios(keypoints)
            
            if not ratios:
                return None
            
            # Compute actual consistency score if references available
            consistency_score = None
            if reference_ratios:
                logger.info(f"Computing ratio consistency - dataset has {len(ratios)} ratios, {len(reference_ratios)} references")
                consistency_score = compute_ratio_consistency(ratios, reference_ratios)
                logger.info(f"Ratio consistency result: {consistency_score}")
            else:
                logger.info("No reference_ratios provided, skipping consistency computation")
            
            return {
                "success": True,
                "ratios": ratios,
                "consistency_score": consistency_score
            }
            
        except Exception as e:
            logger.exception("Keypoint analysis error")
            return None
    
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
    
    def _compute_3d_ratios(self, inference_out: dict) -> dict:
        """
        Compute volumetric ratios from 3D mesh (if mesh vertices available).
        For now, placeholder - returns empty dict.
        """
        # TODO: Implement 3D mesh-based ratio calculation
        # This would use the SMPL vertices to compute limb volumes
        return {}


def compute_beta_consistency(
    dataset_betas: np.ndarray,
    reference_betas_list: list[np.ndarray]
) -> float:
    """
    Compute 3D body shape consistency using SMPL-X beta parameters.
    
    Args:
        dataset_betas: Beta parameters from dataset image (shape: (10,) or (11,))
        reference_betas_list: List of beta parameters from reference images
    
    Returns:
        Consistency score 0-1, where 1 means identical shape
    """
    if not reference_betas_list or dataset_betas is None:
        return None
    
    # Normalize beta dimensions (some may be 10, some 11)
    min_dim = min(len(dataset_betas), min(len(ref) for ref in reference_betas_list))
    dataset_betas_norm = dataset_betas[:min_dim]
    
    # Compute L2 distance to each reference
    distances = []
    for ref_betas in reference_betas_list:
        ref_betas_norm = ref_betas[:min_dim]
        # Euclidean distance in beta space
        dist = np.linalg.norm(dataset_betas_norm - ref_betas_norm)
        distances.append(dist)
    
    # Use minimum distance (closest reference)
    min_distance = min(distances)
    
    # Convert distance to similarity score using exponential decay
    # Typical beta L2 distances range from 0 (identical) to ~5-10 (very different)
    # We use sigma=2.0 so that distance=2 gives ~0.37 similarity
    sigma = 2.0
    similarity = np.exp(-min_distance / sigma)
    
    return float(similarity)


def compute_ratio_consistency(
    dataset_ratios: dict,
    reference_ratios_list: list[dict]
) -> float:
    """
    Compute 2D skeletal ratio consistency.
    
    Args:
        dataset_ratios: Dict of skeletal ratios from dataset image
        reference_ratios_list: List of ratio dicts from reference images
    
    Returns:
        Consistency score 0-1, where 1 means identical proportions
    """
    if not reference_ratios_list or not dataset_ratios:
        return None
    
    # Find common ratio keys across all references and dataset
    common_keys = set(dataset_ratios.keys())
    for ref_ratios in reference_ratios_list:
        common_keys &= set(ref_ratios.keys())
    
    if not common_keys:
        return None
    
    # Compute average absolute percentage differences for each reference
    similarities = []
    
    for ref_ratios in reference_ratios_list:
        diffs = []
        for key in common_keys:
            dataset_val = dataset_ratios[key]
            ref_val = ref_ratios[key]
            
            if ref_val == 0:
                continue
            
            # Percentage difference
            pct_diff = abs(dataset_val - ref_val) / ref_val
            diffs.append(pct_diff)
        
        if diffs:
            # Average percentage difference
            avg_diff = np.mean(diffs)
            # Convert to similarity (0% diff = 100% similarity)
            similarity = np.exp(-avg_diff * 3.0)  # Scale factor 3.0
            similarities.append(similarity)
    
    if not similarities:
        return None
    
    # Use best (maximum) similarity
    return float(max(similarities))


def validate_references(images: dict[str, str], gender: str = "neutral") -> ValidationResult:
    """
    Validate a set of reference images for consistency.
    
    Args:
        images: Dict mapping view type to image path:
            Required:
            - head_front, head_45l, head_45r (Face Analysis)
            - body_front, body_side, body_posterior (Full analysis)
            Optional (improves pose-aware matching):
            - head_90l, head_90r (Full left/right profiles)
            - head_up, head_down (Pitch variation)
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
    optional_head = {
        'head_90l', 'head_90r', 'head_up', 'head_down',
        'head_up_l', 'head_up_r', 'head_down_l', 'head_down_r'
    }
    
    provided = set(images.keys())
    missing_head = required_head - provided
    missing_body = required_body - provided
    
    if missing_head:
        result.warnings.append(ValidationWarning(
            code="MISSING_HEAD_REFS",
            message=f"Missing required head reference images: {missing_head}",
            severity="error"
        ))
    
    if missing_body:
        result.warnings.append(ValidationWarning(
            code="MISSING_BODY_REFS", 
            message=f"Missing required body reference images: {missing_body}",
            severity="error"
        ))
    
    if missing_head or missing_body:
        result.error = "Required reference images missing"
        return result
    
    # === Head Group Analysis (Required + Optional) ===
    head_embeddings = []
    face_embeddings = {}
    face_poses = {}
    
    # Process all provided head references (required + optional)
    all_head_views = required_head | (provided & optional_head)
    
    for view in all_head_views:
        if view not in images:
            continue
            
        face_result = face_analyzer.analyze(images[view])
        if face_result.detected and face_result.embedding is not None:
            head_embeddings.append(face_result.embedding)
            face_embeddings[view] = face_result.embedding
            
            # Store pose for pose-aware matching
            if face_result.pose:
                face_poses[view] = face_result.pose
        else:
            severity = "error" if view in required_head else "warning"
            result.warnings.append(ValidationWarning(
                code="HEAD_FACE_NOT_DETECTED",
                message=f"No face detected in {view}: {face_result.error}",
                severity=severity
            ))
    
    # Store embeddings and poses in result
    result.face_embeddings = face_embeddings
    result.face_poses = face_poses
    
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
                'betas': body_result.betas if body_result.betas is not None else None,
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
    result.face_embeddings = face_embeddings
    result.body_metrics = body_metrics
    result.success = not any(w.severity == "error" for w in result.warnings)
    
    return result


def _check_pose_alignment(pose_data: dict) -> list[ValidationWarning]:
    """Check if body views are at the same scale."""
    warnings = []
    
    # Compare normalized Y-coordinates of key joints (shoulders, hips, knees)
    # This requires pose_data to be populated with normalized keypoints ?
    # Currently our PoseResult stores absolute (pixel) coordinates.
    # We need to normalize by height (ankle to eye/nose) to compare across images of different sizes.
    
    # Simple check: Just report if we have poses for all requested views
    # Real alignment check requires complex normalization
    
    # Compare normalized keypoints for consistency
    # Keypoints to check: shoulders, hips, knees, ankles
    pairs = [
        ('left_shoulder', 'right_shoulder'),
        ('left_hip', 'right_hip'),
        ('left_knee', 'right_knee')
    ]
    
    # We primarily want to check if the 'pose' (stance) is similar enough
    # or if the user is detecting completely different poses (e.g. sitting vs standing)
    # But since we requested standard A-pose/T-pose, we check for symmetry.
    
    for view, keypoints in pose_data.items():
        # Check symmetry
        for left, right in pairs:
            if left in keypoints and right in keypoints:
                ly = keypoints[left]['y']
                ry = keypoints[right]['y']
                # Allow some pixel difference based on image size?
                # Using 5% of image height would be better, but we don't have image height here easily.
                # Just using relative difference between the points.
                diff = abs(ly - ry)
                if diff > 50: # Arbitrary pixel threshold for now (assuming ~1000px height)
                     warnings.append(ValidationWarning(
                        code="POSE_ASYMMETRY",
                        message=f"Significant asymmetry in {view} ({left}/{right} diff: {diff:.0f}px)",
                        severity="warning"
                    ))
    return warnings
    
    return warnings


def _check_volume_consistency(body_metrics: dict) -> list[ValidationWarning]:
    """Check if volume estimates are consistent across views."""
    warnings = []
    
    volumes = [m.get('volume') for m in body_metrics.values() if m.get('volume')]
    if len(volumes) >= 2:
        mean_vol = np.mean(volumes)
        max_var = max(abs(v - mean_vol) / mean_vol for v in volumes)
        if max_var > 0.10:  # 10% variance threshold
            # Check if we are in degraded mode - volume estimates from 2D are noisy
            # So we increase threshold or skip warning if strictly 2D
            severity = "warning"
            
            warnings.append(ValidationWarning(
                code="VOLUME_INCONSISTENCY",
                message=f"Volume estimates vary by {max_var*100:.1f}% across views",
                severity=severity
            ))
    
    return warnings


def _check_ratio_consistency(body_metrics: dict) -> list[ValidationWarning]:
    """Check if limb ratios are consistent across body views."""
    warnings = []
    
    if not body_metrics:
        return warnings
    
    # Extract ratios from each view's metrics
    # New format: {view: {"metrics_3d": {...}, "metrics_2d": {...}, "preferred": "3d"}}
    # Old format: {view: {"ratios": {...}, "degraded_mode": bool}}
    
    ratio_by_type = {}  # ratio_name -> [values]
    
    for view, metrics in body_metrics.items():
        # Handle new dual-metrics structure
        if 'metrics_3d' in metrics or 'metrics_2d' in metrics:
            # Use preferred metric or fallback to 3d then 2d
            preferred = metrics.get('preferred', '3d')
            
            if preferred == '3d' and 'metrics_3d' in metrics:
                ratios = metrics['metrics_3d'].get('ratios', {})
            elif preferred == '2d' and 'metrics_2d' in metrics:
                ratios = metrics['metrics_2d'].get('ratios', {})
            elif 'metrics_3d' in metrics:
                ratios = metrics['metrics_3d'].get('ratios', {})
            elif 'metrics_2d' in metrics:
                ratios = metrics['metrics_2d'].get('ratios', {})
            else:
                continue
        # Handle old flat structure
        elif 'ratios' in metrics:
            ratios = metrics['ratios']
        else:
            continue
        
        # Collect ratio values
        for ratio_name, ratio_value in ratios.items():
            if isinstance(ratio_value, (int, float)):
                if ratio_name not in ratio_by_type:
                    ratio_by_type[ratio_name] = []
                ratio_by_type[ratio_name].append(ratio_value)
    
    # Check consistency (coefficient of variation < 0.15)
    for ratio_name, values in ratio_by_type.items():
        if len(values) >= 2:
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val != 0 else 0
            
            if cv > 0.15:
                warnings.append(ValidationWarning(
                    code="RATIO_INCONSISTENCY",
                    message=f"Inconsistent {ratio_name} across views (CV={cv:.2f})",
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
