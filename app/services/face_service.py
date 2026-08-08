"""
Core deep learning face detection and embedding service.

This module wraps InsightFace's FaceAnalysis pipeline which uses two deep learning models:

1. RetinaFace (det_10g.onnx) — A single-stage dense face detector based on a
   Feature Pyramid Network. It outputs bounding boxes, confidence scores, and
   5 facial landmarks (eyes, nose, mouth corners) for each detected face.

2. ArcFace ResNet-50 (w600k_r50.onnx) — A ResNet-50 backbone trained with
   ArcFace (Additive Angular Margin) loss on 600K identities. It produces a
   512-dimensional embedding vector for each aligned face. The angular margin
   penalty during training forces the network to learn embeddings where faces
   of the same person are angularly close and different people are far apart
   on a unit hypersphere.

Pipeline for each image:
    Raw image → RetinaFace detection → landmark-based alignment → ArcFace embedding → L2-normalized 512-dim vector

The resulting embeddings can be compared using cosine similarity (which equals
dot product for L2-normalized vectors) to determine if two faces belong to the
same person.
"""

import logging
import threading

import numpy as np
import insightface
from insightface.app import FaceAnalysis

from app.config import get_settings

logger = logging.getLogger(__name__)


class FaceService:
    """
    Singleton-style service for face detection and embedding generation.

    The InsightFace model is loaded lazily on first use and cached for all
    subsequent calls. Loading is thread-safe via a lock.
    """

    def __init__(self):
        self._model: FaceAnalysis | None = None
        self._lock = threading.Lock()
        self._settings = get_settings()

    def _ensure_model_loaded(self) -> None:
        """Load the InsightFace model pack if not already loaded."""
        if self._model is not None:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            logger.info(f"Loading InsightFace model pack: {self._settings.model_pack_name}")

            self._model = FaceAnalysis(
                name=self._settings.model_pack_name,
                providers=["CPUExecutionProvider"],
            )

            det_size = self._settings.detection_size
            self._model.prepare(
                ctx_id=-1,  # -1 = CPU
                det_size=(det_size, det_size),
                det_thresh=self._settings.detection_threshold,
            )

            logger.info(
                f"Model loaded successfully. Detection size: {det_size}x{det_size}, "
                f"threshold: {self._settings.detection_threshold}"
            )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._settings.model_pack_name

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        """
        Detect all faces in an image using the RetinaFace deep learning detector.

        Args:
            image: BGR numpy array (OpenCV format)

        Returns:
            List of dicts, each containing:
                - face_index: int
                - bbox: dict with x, y, width, height
                - confidence: float
                - landmarks: list of 5 [x, y] points
                - _insightface_obj: raw InsightFace face object (for embedding extraction)
        """
        self._ensure_model_loaded()

        faces = self._model.get(image)

        results = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            results.append({
                "face_index": i,
                "bbox": {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                },
                "confidence": float(face.det_score),
                "landmarks": face.kps.tolist() if face.kps is not None else [],
                "_insightface_obj": face,
            })

        # Sort by confidence descending so the most prominent face is first
        results.sort(key=lambda f: f["confidence"], reverse=True)
        for i, face in enumerate(results):
            face["face_index"] = i

        return results

    def get_embedding(self, face_obj) -> np.ndarray:
        """
        Extract the 512-dimensional ArcFace embedding for a detected face.

        The InsightFace pipeline handles face alignment internally: it uses the
        5 detected landmarks to compute a similarity transform that aligns the
        face crop to a canonical 112x112 template before passing it through
        the ArcFace ResNet-50 network.

        The output embedding is already L2-normalized by InsightFace, meaning
        its magnitude is 1.0 and cosine similarity equals the dot product.

        Args:
            face_obj: InsightFace face object from detect_faces

        Returns:
            512-dimensional numpy array (L2-normalized)
        """
        embedding = face_obj.normed_embedding

        # Verify normalization — should be ~1.0
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) > 0.01:
            logger.warning(f"Embedding norm {norm:.4f} deviates from 1.0, re-normalizing")
            embedding = embedding / norm

        return embedding

    def process_image(self, image: np.ndarray) -> dict:
        """
        Full pipeline: detect all faces in an image and generate embeddings for each.

        This is the primary method used by the /api/face/process-photo endpoint.

        Args:
            image: BGR numpy array

        Returns:
            dict with:
                - faces_detected: int
                - image_width, image_height: int
                - model_name: str
                - faces: list of face results with embeddings
        """
        height, width = image.shape[:2]

        detected = self.detect_faces(image)

        faces_with_embeddings = []
        for face_data in detected:
            face_obj = face_data.pop("_insightface_obj")
            embedding = self.get_embedding(face_obj)

            face_data["embedding"] = embedding.tolist()
            face_data["embedding_dim"] = len(embedding)
            faces_with_embeddings.append(face_data)

        return {
            "faces_detected": len(faces_with_embeddings),
            "image_width": width,
            "image_height": height,
            "model_name": self._settings.model_pack_name,
            "faces": faces_with_embeddings,
        }

    @staticmethod
    def compute_similarity(embedding_a: list[float], embedding_b: list[float]) -> dict:
        """
        Compare two face embeddings using cosine similarity and Euclidean distance.

        For L2-normalized embeddings (as produced by ArcFace):
            cosine_similarity = dot(A, B)
            euclidean_distance = sqrt(2 - 2 * cosine_similarity)

        A default threshold of 0.4 is used for the is_same_person flag.
        This is a conservative starting point — the actual production threshold
        will be experimentally determined in Phase 10 using precision/recall
        analysis on a validation dataset.

        Args:
            embedding_a: First 512-dim embedding
            embedding_b: Second 512-dim embedding

        Returns:
            dict with similarity, distance, is_same_person, threshold_used
        """
        a = np.array(embedding_a, dtype=np.float32)
        b = np.array(embedding_b, dtype=np.float32)

        # Normalize in case embeddings were serialized/deserialized with precision loss
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        cosine_sim = float(np.dot(a, b))
        euclidean_dist = float(np.sqrt(np.maximum(2.0 - 2.0 * cosine_sim, 0.0)))

        default_threshold = 0.4

        return {
            "similarity": round(cosine_sim, 6),
            "distance": round(euclidean_dist, 6),
            "is_same_person": cosine_sim >= default_threshold,
            "threshold_used": default_threshold,
        }


# Module-level singleton so the model is loaded once and shared across requests
face_service = FaceService()
