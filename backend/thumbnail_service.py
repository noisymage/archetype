"""
Thumbnail generation and caching service.

Provides file-based caching for thumbnails with on-demand generation.
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent / ".cache" / "thumbnails"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Maximum thumbnail dimension
MAX_THUMBNAIL_SIZE = 256


def get_thumbnail_path(image_path: str, size: int = MAX_THUMBNAIL_SIZE) -> Path:
    """Get the cache path for a thumbnail."""
    # Create hash from absolute path + size
    path_hash = hashlib.md5(f"{image_path}:{size}".encode()).hexdigest()
    return CACHE_DIR / f"{path_hash}.jpg"


def generate_thumbnail(image_path: str, size: int = MAX_THUMBNAIL_SIZE) -> Optional[str]:
    """
    Generate a thumbnail for an image, returning the cache path.
    
    Uses file-based caching - returns cached version if exists.
    """
    source = Path(image_path)
    if not source.exists():
        logger.error(f"Source image not found: {image_path}")
        return None
    
    cache_path = get_thumbnail_path(image_path, size)
    
    # Return cached version if exists and is newer than source
    if cache_path.exists():
        if cache_path.stat().st_mtime >= source.stat().st_mtime:
            return str(cache_path)
    
    try:
        with Image.open(source) as img:
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if img.mode in ('RGBA', 'P', 'LA'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new dimensions maintaining aspect ratio
            ratio = min(size / img.width, size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            
            # High-quality resize
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save as JPEG with good quality
            img_resized.save(cache_path, 'JPEG', quality=85, optimize=True)
            
            logger.debug(f"Generated thumbnail: {cache_path}")
            return str(cache_path)
            
    except Exception as e:
        logger.exception(f"Failed to generate thumbnail for {image_path}: {e}")
        return None


def clear_thumbnail_cache():
    """Clear all cached thumbnails."""
    count = 0
    for thumb in CACHE_DIR.glob("*.jpg"):
        try:
            thumb.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {thumb}: {e}")
    logger.info(f"Cleared {count} cached thumbnails")
    return count


def get_cache_size() -> int:
    """Get total size of thumbnail cache in bytes."""
    total = 0
    for thumb in CACHE_DIR.glob("*.jpg"):
        total += thumb.stat().st_size
    return total
