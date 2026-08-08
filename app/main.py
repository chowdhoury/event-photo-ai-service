"""
Event Photo AI — Face Processing API

FastAPI application exposing deep learning face detection and embedding
generation endpoints. This service is intended to be called by the Node.js
backend, not directly by end users.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.schemas import (
    ImageProcessingResponse,
    FaceComparisonRequest,
    FaceComparisonResponse,
    HealthResponse,
    ErrorResponse,
)
from app.services.face_service import face_service
from app.utils.image_utils import validate_image, decode_image, resize_if_needed

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the deep learning models on startup so the first request isn't slow."""
    logger.info("Starting AI service — loading face detection and recognition models...")
    face_service._ensure_model_loaded()
    logger.info("Models loaded and ready.")
    yield
    logger.info("AI service shutting down.")


app = FastAPI(
    title="Event Photo AI — Face Processing Service",
    description=(
        "Deep learning face detection (RetinaFace) and face embedding generation "
        "(ArcFace ResNet-50) for event photo matching."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=face_service.is_loaded,
        model_name=face_service.model_name,
        version="0.1.0",
    )


# ---------------------------------------------------------------------------
# Face detection only (no embeddings)
# ---------------------------------------------------------------------------

@app.post("/api/face/detect", responses={400: {"model": ErrorResponse}})
async def detect_faces(file: UploadFile = File(..., description="Image file (JPEG, PNG, or WebP)")):
    """
    Detect all faces in an uploaded image using the RetinaFace deep learning detector.
    Returns bounding boxes, confidence scores, and facial landmarks.
    Does NOT compute embeddings — use /api/face/process-photo for that.
    """
    file_bytes = await file.read()

    is_valid, error_msg = validate_image(
        file_bytes,
        content_type=file.content_type,
        max_size_mb=settings.max_file_size_mb,
        max_dimension=settings.max_image_dimension,
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    image = decode_image(file_bytes)
    image = resize_if_needed(image, settings.max_image_dimension)

    detected = face_service.detect_faces(image)

    # Strip internal InsightFace objects before returning
    results = []
    for face in detected:
        face.pop("_insightface_obj", None)
        results.append(face)

    height, width = image.shape[:2]

    return {
        "success": True,
        "faces_detected": len(results),
        "image_width": width,
        "image_height": height,
        "faces": results,
    }


# ---------------------------------------------------------------------------
# Full pipeline: detect faces + generate embeddings
# ---------------------------------------------------------------------------

@app.post(
    "/api/face/process-photo",
    response_model=ImageProcessingResponse,
    responses={400: {"model": ErrorResponse}},
)
async def process_photo(file: UploadFile = File(..., description="Image file (JPEG, PNG, or WebP)")):
    """
    Full deep learning pipeline:
    1. Validate and decode the uploaded image
    2. Detect all faces using RetinaFace
    3. Align each face using detected landmarks
    4. Generate a 512-dimensional ArcFace embedding for each face
    5. Return all face data including embeddings

    Each face in a group photo gets its own embedding vector. These vectors
    are later stored in pgvector for similarity search when a guest uploads
    their selfie.
    """
    file_bytes = await file.read()

    is_valid, error_msg = validate_image(
        file_bytes,
        content_type=file.content_type,
        max_size_mb=settings.max_file_size_mb,
        max_dimension=settings.max_image_dimension,
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    image = decode_image(file_bytes)
    image = resize_if_needed(image, settings.max_image_dimension)

    result = face_service.process_image(image)

    return ImageProcessingResponse(
        success=True,
        faces_detected=result["faces_detected"],
        image_width=result["image_width"],
        image_height=result["image_height"],
        model_name=result["model_name"],
        faces=result["faces"],
    )


# ---------------------------------------------------------------------------
# Compare two face embeddings
# ---------------------------------------------------------------------------

@app.post(
    "/api/face/compare",
    response_model=FaceComparisonResponse,
    responses={400: {"model": ErrorResponse}},
)
async def compare_faces(request: FaceComparisonRequest):
    """
    Compare two 512-dimensional face embedding vectors.

    Computes cosine similarity (range: -1 to 1, higher = more similar) and
    Euclidean distance. For L2-normalized ArcFace embeddings, cosine similarity
    equals the dot product.

    The is_same_person flag uses a default threshold of 0.4 — this will be
    experimentally tuned in Phase 10 using precision/recall analysis.
    """
    if len(request.embedding_a) != 512 or len(request.embedding_b) != 512:
        raise HTTPException(
            status_code=400,
            detail=f"Embeddings must be 512-dimensional. "
                   f"Got {len(request.embedding_a)} and {len(request.embedding_b)}.",
        )

    result = face_service.compute_similarity(request.embedding_a, request.embedding_b)

    return FaceComparisonResponse(**result)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)},
    )
