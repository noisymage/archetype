"""
AdaFace inference utilities.

Vendored and simplified from: https://github.com/mk-minchul/AdaFace
Provides model loading and feature extraction.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from pathlib import Path

from .net import IR_101

logger = logging.getLogger(__name__)

# Model download URL (Google Drive file ID for ir_101 pretrained on WebFace4M)
ADAFACE_MODEL_URL = "https://drive.google.com/uc?id=1dswnavflETcnAuplZj1IOKKP0eM8ITgT"
ADAFACE_MODEL_FILENAME = "adaface_ir101_webface4m.ckpt"


def download_model(model_dir: Path) -> Path:
    """
    Download AdaFace pretrained model if not present.
    
    Args:
        model_dir: Directory to save the model
        
    Returns:
        Path to the downloaded model file
    """
    model_path = model_dir / ADAFACE_MODEL_FILENAME
    
    if model_path.exists():
        return model_path
    
    logger.info(f"Downloading AdaFace model to {model_path}...")
    
    try:
        import gdown
        model_dir.mkdir(parents=True, exist_ok=True)
        gdown.download(ADAFACE_MODEL_URL, str(model_path), quiet=False)
        logger.info("AdaFace model downloaded successfully")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download AdaFace model: {e}")
        raise


def load_pretrained_model(model_path: str, device: str = "cpu"):
    """
    Load pretrained AdaFace model.
    
    Args:
        model_path: Path to the .ckpt file
        device: "cpu", "cuda", or "mps"
        
    Returns:
        Loaded model in eval mode
    """
    model = IR_101()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from training checkpoint)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


def to_input(pil_rgb_image):
    """
    Convert PIL RGB image to AdaFace input tensor.
    
    AdaFace expects BGR 112x112 images normalized with mean=0.5, std=0.5
    
    Args:
        pil_rgb_image: PIL Image in RGB format, should be 112x112
        
    Returns:
        Tensor of shape (1, 3, 112, 112) in BGR order
    """
    np_img = np.array(pil_rgb_image)
    
    # Convert RGB to BGR
    bgr_img = np_img[:, :, ::-1].copy()
    
    # Normalize to [-1, 1] (mean=0.5, std=0.5 on [0,1] scale)
    bgr_img = bgr_img.astype(np.float32) / 255.0
    bgr_img = (bgr_img - 0.5) / 0.5
    
    # HWC to CHW
    tensor = torch.from_numpy(bgr_img.transpose(2, 0, 1))
    
    # Add batch dimension
    return tensor.unsqueeze(0)


@torch.no_grad()
def extract_embedding(model, aligned_face, device: str = "cpu") -> np.ndarray:
    """
    Extract face embedding using AdaFace.
    
    Args:
        model: Loaded AdaFace model
        aligned_face: PIL Image (RGB, 112x112) or numpy array
        device: Compute device
        
    Returns:
        512-dimensional normalized embedding
    """
    from PIL import Image
    
    # Convert numpy to PIL if needed
    if isinstance(aligned_face, np.ndarray):
        aligned_face = Image.fromarray(aligned_face)
    
    # Ensure correct size
    if aligned_face.size != (112, 112):
        aligned_face = aligned_face.resize((112, 112))
    
    # Convert to input tensor  
    input_tensor = to_input(aligned_face).to(device)
    
    # Extract features
    embedding = model(input_tensor)
    
    return embedding.cpu().numpy().flatten()
