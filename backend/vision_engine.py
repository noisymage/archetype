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
# Internal structure is often: backend/smplest_x_lib/ ...
SMPLEST_X_PATH = Path(__file__).parent / "smplest_x_lib"

if SMPLEST_X_PATH.exists():
    # 1. Add the lib root so "from smplest_x_lib.demo import ..." works if needed (less likely)
    if str(SMPLEST_X_PATH) not in sys.path:
        sys.path.insert(0, str(SMPLEST_X_PATH))
    
    # 2. CRITICAL: Add the library root so its internal relative imports resolve 
    # (some repos assume they are the root)
    if str(SMPLEST_X_PATH.resolve()) not in sys.path:
        sys.path.insert(0, str(SMPLEST_X_PATH.resolve()))

    # 3. Add 'main' if that is where config stays (common in some research code)
    if str((SMPLEST_X_PATH / "main").resolve()) not in sys.path:
        sys.path.insert(0, str((SMPLEST_X_PATH / "main").resolve()))


# Configure logging
logger = logging.getLogger(__name__)

SMPLEST_X_AVAILABLE = False
try:
    # Try importing essential SMPLest-X modules
    # We do this after path injection
    import config as smpl_config_module 
    from human_models.human_models import SMPLX
    from models.SMPLest_X import get_model
    from utils.data_utils import load_img, process_bbox, generate_patch_image
    # main is in sys.path, so imports work directly
    from config import Config as SMPLConfig
    SMPLEST_X_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SMPLest-X imports failed despite path injection: {e}")
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


@dataclass 
class ValidationResult:
    """Result from reference image validation."""
    success: bool = False
    degraded_mode: bool = True
    warnings: list[ValidationWarning] = field(default_factory=list)
    master_embedding: Optional[np.ndarray] = None
    face_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    body_metrics: Optional[dict] = None
    error: Optional[str] = None


class SMPLestXWrapper:
    """
    Wrapper for SMPLest-X inference to handle complex config and model setup.
    """
    def __init__(self, device: str):
        self.device = device
        self.cfg = None
        self.model = None
        self.smpl_x = None
        self.initialized = False
        
    def initialize(self) -> bool:
        if self.initialized:
            return True
            
        if not SMPLEST_X_AVAILABLE:
            return False
            
        try:
            # Paths
            ckpt_path = MODELS_DIR / "smplx/pretrained/smplest_x_h.pth.tar"
            human_model_path = MODELS_DIR / "smplx/body_models"
            
            if not ckpt_path.exists():
                logger.error(f"SMPLest-X checkpoint missing: {ckpt_path}")
                return False
                
            # Scaffold Config (mimic config_smplest_x_h.py)
            conf_dict = {
                "model": {
                    'model_type': 'vit_huge',
                    "pretrained_model_path": str(ckpt_path),
                    "human_model_path": str(human_model_path),
                    'encoder_config': {
                        'num_classes': 80, 'task_tokens_num': 80, 'img_size': (256, 192),
                        'patch_size': 16, 'embed_dim': 1280, 'depth': 32, 'num_heads': 16,
                        'ratio': 1, 'use_checkpoint': False, 'mlp_ratio': 4, 'qkv_bias': True, 'drop_path_rate': 0.55
                    },
                    'decoder_config': {'feat_dim': 1280, "dim_out": 512, 'task_tokens_num': 80},
                    'input_img_shape': (512, 384),
                    'input_body_shape': (256, 192),
                    'output_hm_shape': (16, 16, 12),
                    'focal': (5000, 5000), 'princpt': (192 / 2, 256 / 2),
                    'camera_3d_size': 2.5,
                },
                "test": {"test_batch_size": 1},
                "data": {"testset": "custom"}
            }
            
            self.cfg = SMPLConfig(conf_dict)
            
            # Init Human Model
            self.smpl_x = SMPLX(str(human_model_path))
            
            # Init Network
            logger.info("Loading SMPLest-X model (this may take a moment)...")
            self.model = get_model(self.cfg, 'test')
            
            # Load Checkpoint
            if self.device == 'mps':
                 # MPS sometimes needs CPU load -> move
                 ckpt = torch.load(str(ckpt_path), map_location='cpu')
            else:
                 ckpt = torch.load(str(ckpt_path), map_location='cpu')
                 
            # State dict processing (from Tester._make_model)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['network'].items():
                if 'module' not in k: k = 'module.' + k
                k = k.replace('backbone', 'encoder').replace('body_rotation_net', 'body_regressor').replace(
                    'hand_rotation_net', 'hand_regressor')
                new_state_dict[k] = v
                
            # Load with strict=False as per their code
            self.model.load_state_dict(new_state_dict, strict=False)
            
            # Move to device
            if self.device == 'cuda':
                self.model = self.model.cuda()
            elif self.device == 'mps':
                self.model = self.model.to('mps')
            else:
                self.model = self.model.cpu()
                
            self.model.eval()
            self.initialized = True
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize SMPLest-X: {e}")
            return False

    def run_inference(self, image: np.ndarray, bbox: list) -> dict:
        """
        Run inference on a single image with a given bounding box.
        bbox: [x1, y1, x2, y2]
        """
        if not self.initialized:
            return None
            
        try:
            import torch
            from torchvision import transforms
            
            # Prepare bounding box (xyxy -> xywh)
            # Their code expects: box_xywh[2] = width, box_xywh[3] = height
            bbox_xywh = np.array([bbox[0], bbox[1], abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1])])
            
            # Preprocess using their utils
            original_img_height, original_img_width = image.shape[:2]
            proc_bbox = process_bbox(bbox_xywh, original_img_width, original_img_height, 
                                     input_img_shape=self.cfg.model.input_img_shape)
            
            img_patch, _, _ = generate_patch_image(cvimg=image.copy(), bbox=proc_bbox, 
                                                   scale=1.0, rot=0.0, do_flip=False, 
                                                   out_shape=self.cfg.model.input_img_shape)
            
            # Transform
            transform = transforms.ToTensor()
            img_tensor = transform(img_patch.astype(np.float32))/255
            img_tensor = img_tensor.unsqueeze(0) # Batch dim
            
            if self.device == 'cuda':
                img_tensor = img_tensor.cuda()
            elif self.device == 'mps':
                 img_tensor = img_tensor.to('mps')
            
            # Run Model
            with torch.no_grad():
                inputs = {'img': img_tensor}
                out = self.model(inputs, {}, {}, 'test')
            
            # Extract Results
            betas = out['smplx_shape'].reshape(-1, 10).cpu().numpy()[0]
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
            
            return {
                'betas': betas.tolist(),
                'mesh': mesh, 
                'volume_proxy': self._calc_volume(mesh)
            }
            
        except Exception as e:
            logger.exception(f"Inference run failed: {e}")
            return None

    def _calc_volume(self, mesh_vertices):
        """Estimate volume from mesh vertices using Convex Hull."""
        try:
            # Load convex hull
            hull = spatial.ConvexHull(mesh_vertices)
            return float(hull.volume * 1000) # Convert m^3 to liters (assuming vertices in meters) 
            # Note: SMPL-X usually output in meters. 1 m^3 = 1000 Liters.
            # If vertices are not in meters, this scaling factor needs adjustment.
            # Usually SMPL templates are unit height ~1.7 range.
        except Exception as e:
            logger.warning(f"Volume calculation failed: {e}")
            # Fallback to bbox
            try:
                min_coords = np.min(mesh_vertices, axis=0)
                max_coords = np.max(mesh_vertices, axis=0)
                dims = max_coords - min_coords 
                vol = dims[0] * dims[1] * dims[2]
                return float(vol * 1000)
            except:
                return 0.0


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
        self._smplx_wrapper = None # Lazy load SMPLest-X 
        
        # State tracking
        self._face_loaded = False
        self._pose_loaded = False
        self._smplx_loaded = False
        
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
        return self._smplx_loaded and self._smplx_wrapper is not None and self._smplx_wrapper.initialized
    
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

    
    def load_mesh_model(self, gender: str = "neutral") -> bool:
        """
        Initialize SMPLest-X model.
        """
        if self._smplx_loaded and self._smplx_wrapper:
            return self.smplx_available
            
        # try:
        try:
            if not self._smplx_wrapper:
                self._smplx_wrapper = SMPLestXWrapper(self._device)
                
            if self._smplx_wrapper.initialize():
                self._smplx_loaded = True
                self._current_gender = "neutral" 
                logger.info("SMPLest-X model loaded successfully")
                return True
            else:
                logger.warning("SMPLest-X initialization failed, degrading to 2D")
                self._smplx_loaded = True # Mark as "attempted/loaded" but in degraded state
                return False
                
        except Exception as e:
            logger.error(f"Failed to load SMPLest-X: {e}")
            self._smplx_loaded = True
            return False
    
    def _unload_mesh_model(self):
        """Unload SMPL-X model."""
        if self._smplx_wrapper is not None:
            del self._smplx_wrapper
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
        Analyze body shape and proportions using both 3D and 2D methods.
        
        Args:
            image_path: Absolute path to image file.
            keypoints: Pre-computed keypoints (optional).
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
        metrics_2d = None
        preferred = "none"
        
        # Attempt 3D analysis
        smplx_loaded = self._manager.load_mesh_model(gender)
        if smplx_loaded and self._manager._smplx_wrapper is not None:
            try:
                metrics_3d = self._analyze_with_smplx(image_path, keypoints)
                if metrics_3d and metrics_3d.get('success'):
                    preferred = "3d"
            except Exception as e:
                logger.warning(f"3D analysis failed, continuing with 2D: {e}")
        
        # Always attempt 2D analysis if keypoints available
        if keypoints:
            try:
                metrics_2d = self._analyze_with_keypoints(keypoints)
                if metrics_2d and metrics_2d.get('success'):
                    if preferred == "none":
                        preferred = "2d"
            except Exception as e:
                logger.warning(f"2D analysis failed: {e}")
        
        # Populate result
        result.analyzed = (metrics_3d is not None or metrics_2d is not None)
        result.degraded_mode = (preferred == "2d")
        
        # Store dual metrics in ratios field (will be restructured in batch_processor)
        result.ratios = {
            "metrics_3d": metrics_3d,
            "metrics_2d": metrics_2d,
            "preferred": preferred
        }
        
        return result
    
    def _analyze_with_smplx(self, image_path: str, keypoints: Optional[dict] = None) -> Optional[dict]:
        """
        Full 3D analysis using SMPLest-X.
        
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
                
            # Get bounding box from pose detection
            if not self._manager._pose_loaded:
                self._manager.load_pose_model()
            
            pose_res = self._manager._pose_model(image_path, verbose=False)[0]
            if not pose_res.boxes or len(pose_res.boxes) == 0:
                return None
                 
            # Best person
            boxes = pose_res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(areas.argmax())
            bbox = boxes[best_idx].tolist()
            
            # Run SMPLest-X inference
            inference_out = wrapper.run_inference(img, bbox)
            
            if inference_out:
                # Calculate 3D-based ratios from mesh if available
                ratios = self._compute_3d_ratios(inference_out) if keypoints else {}
                
                return {
                    "success": True,
                    "betas": inference_out['betas'],
                    "volume": inference_out.get('volume_proxy'),
                    "ratios": ratios,
                    "consistency_score": 0.85  # Placeholder - implement proper consistency metric
                }
            
            return None
            
        except Exception as e:
            logger.exception(f"SMPLest-X analysis error: {e}")
            return None
    
    def _analyze_with_keypoints(self, keypoints: dict) -> Optional[dict]:
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
            
            return {
                "success": True,
                "ratios": ratios,
                "consistency_score": 0.75  # Placeholder - 2D is less reliable
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
    face_embeddings = {}
    for view in ['head_front', 'head_45l', 'head_45r']:
        face_result = face_analyzer.analyze(images[view])
        if face_result.detected and face_result.embedding is not None:
            head_embeddings.append(face_result.embedding)
            face_embeddings[view] = face_result.embedding
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
