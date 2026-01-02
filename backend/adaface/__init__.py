"""
AdaFace - Quality Adaptive Margin for Face Recognition

Vendored inference code from: https://github.com/mk-minchul/AdaFace
"""

from .inference import load_pretrained_model, extract_embedding, download_model
from .face_align import align_face, get_mtcnn
from .net import IR_101, IR_50

__all__ = [
    'load_pretrained_model',
    'extract_embedding', 
    'download_model',
    'align_face',
    'get_mtcnn',
    'IR_101',
    'IR_50'
]
