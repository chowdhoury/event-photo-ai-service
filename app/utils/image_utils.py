import io
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MIME_EXTENSION_MAP = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/webp": [".webp"],
}


def validate_image(
    file_bytes: bytes,
    content_type: str | None,
    max_size_mb: int = 10,
    max_dimension: int = 4096,
) -> tuple[bool, str | None]:
    """
    Validates an uploaded image file.

    Returns (is_valid, error_message).
    """
    max_size_bytes = max_size_mb * 1024 * 1024

    if len(file_bytes) > max_size_bytes:
        return False, f"File size ({len(file_bytes) / 1024 / 1024:.1f}MB) exceeds maximum ({max_size_mb}MB)"

    if len(file_bytes) < 100:
        return False, "File is too small to be a valid image"

    if content_type and content_type not in ALLOWED_MIME_TYPES:
        return False, f"Unsupported image type: {content_type}. Allowed: JPEG, PNG, WebP"

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception:
        return False, "File is not a valid image or is corrupted"

    # Re-open after verify() since verify() can invalidate the image object
    try:
        img = Image.open(io.BytesIO(file_bytes))
        width, height = img.size
    except Exception:
        return False, "Failed to read image dimensions"

    if width > max_dimension or height > max_dimension:
        return False, (
            f"Image dimensions ({width}x{height}) exceed maximum ({max_dimension}x{max_dimension}). "
            f"The image will be resized automatically if processing is requested."
        )

    if width < 20 or height < 20:
        return False, f"Image dimensions ({width}x{height}) are too small for face detection"

    return True, None


def decode_image(file_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into an OpenCV BGR numpy array.
    InsightFace expects BGR format (standard OpenCV convention).
    """
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image. The file may be corrupted or in an unsupported format.")

    return image


def resize_if_needed(image: np.ndarray, max_dimension: int = 4096) -> np.ndarray:
    """
    Downscales an image if either dimension exceeds max_dimension,
    preserving aspect ratio.
    """
    height, width = image.shape[:2]

    if width <= max_dimension and height <= max_dimension:
        return image

    scale = max_dimension / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized
