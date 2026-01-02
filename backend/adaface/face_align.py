"""
Face alignment utilities using MTCNN from facenet-pytorch.

Provides face detection and alignment to 112x112 for AdaFace.
"""
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded MTCNN instance
_mtcnn = None


def get_mtcnn(device: str = "cpu"):
    """Get or create MTCNN detector."""
    global _mtcnn
    
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            _mtcnn = MTCNN(
                image_size=112,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],  # Slightly lower for difficult poses
                factor=0.709,
                post_process=True,  # Returns PIL Image
                device=device
            )
            logger.info("MTCNN loaded for AdaFace alignment")
        except ImportError:
            logger.error("facenet-pytorch not installed. Run: pip install facenet-pytorch")
            raise
    
    return _mtcnn


def align_face(image, device: str = "cpu"):
    """
    Detect and align face to 112x112 for AdaFace.
    
    Args:
        image: PIL Image, numpy array (BGR or RGB), or path string
        device: Compute device
        
    Returns:
        PIL Image (RGB, 112x112) if face detected, None otherwise
    """
    import cv2
    
    # Handle different input types
    if isinstance(image, str):
        # Path string
        img = cv2.imread(image)
        if img is None:
            return None
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
    elif isinstance(image, np.ndarray):
        # Check if BGR (OpenCV default) or RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img = image
        pil_img = Image.fromarray(img)
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        logger.warning(f"Unsupported image type: {type(image)}")
        return None
    
    # Detect and align
    mtcnn = get_mtcnn(device)
    
    try:
        # MTCNN returns aligned face or None
        aligned = mtcnn(pil_img)
        
        if aligned is None:
            return None
        
        # Convert tensor to PIL if needed
        if hasattr(aligned, 'cpu'):
            # It's a tensor, convert to numpy then PIL
            aligned_np = aligned.permute(1, 2, 0).cpu().numpy()
            # Denormalize from [-1, 1] to [0, 255]
            aligned_np = ((aligned_np + 1) * 127.5).astype(np.uint8)
            return Image.fromarray(aligned_np)
        
        return aligned
        
    except Exception as e:
        logger.debug(f"MTCNN face detection failed: {e}")
        return None
