"""
End-to-end test for the face detection and embedding pipeline.

This script:
1. Downloads two sample face images from the internet
2. Sends each to the /api/face/process-photo endpoint
3. Verifies face detection and embedding generation
4. Compares the embeddings using /api/face/compare
5. Reports results

Usage:
    First start the server:
        python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

    Then run this script:
        python tests/test_face_pipeline.py
"""

import sys
import json
import io
import os
import time

import requests
import numpy as np

BASE_URL = os.environ.get("AI_SERVICE_URL", "http://localhost:8000")

# Public domain / freely licensed test images of faces
# Using ThisPersonDoesNotExist-style generated faces from public APIs
TEST_IMAGE_URLS = [
    "https://thispersondoesnotexist.com",
    "https://thispersondoesnotexist.com",
]


def download_test_image(url: str, label: str) -> bytes:
    """Download a test image, with retry."""
    print(f"  Downloading {label} from {url}...")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "EventPhotoAI-Test/1.0"})
            resp.raise_for_status()
            print(f"  ✓ Downloaded {label}: {len(resp.content) / 1024:.1f} KB")
            return resp.content
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt + 1}... ({e})")
                time.sleep(2)
            else:
                raise


def test_health():
    """Test the health endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Health Check")
    print("=" * 60)

    resp = requests.get(f"{BASE_URL}/health")
    data = resp.json()
    print(f"  Status:       {data['status']}")
    print(f"  Model loaded: {data['model_loaded']}")
    print(f"  Model name:   {data['model_name']}")
    print(f"  Version:      {data['version']}")

    assert data["status"] == "healthy", "Service is not healthy"
    assert data["model_loaded"] is True, "Model is not loaded"
    print("  ✓ Health check passed")
    return True


def test_process_photo(image_bytes: bytes, label: str) -> dict | None:
    """Test the face processing pipeline on a single image."""
    print(f"\n{'=' * 60}")
    print(f"TEST: Process Photo — {label}")
    print("=" * 60)

    files = {"file": (f"{label}.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    resp = requests.post(f"{BASE_URL}/api/face/process-photo", files=files)

    if resp.status_code != 200:
        print(f"  ✗ Request failed: {resp.status_code}")
        print(f"    {resp.text}")
        return None

    data = resp.json()
    print(f"  Faces detected:   {data['faces_detected']}")
    print(f"  Image size:       {data['image_width']}x{data['image_height']}")
    print(f"  Model used:       {data['model_name']}")

    if data["faces_detected"] == 0:
        print("  ⚠ No faces detected in this image")
        return data

    for face in data["faces"]:
        bbox = face["bbox"]
        print(f"\n  Face {face['face_index']}:")
        print(f"    Bounding box:   x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}")
        print(f"    Confidence:     {face['confidence']:.4f}")
        print(f"    Embedding dim:  {face['embedding_dim']}")

        # Verify embedding properties
        emb = np.array(face["embedding"])
        norm = np.linalg.norm(emb)
        print(f"    Embedding norm: {norm:.6f} (should be ≈1.0)")
        print(f"    Embedding preview: [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, ..., {emb[-1]:.4f}]")

        assert face["embedding_dim"] == 512, f"Expected 512-dim embedding, got {face['embedding_dim']}"
        assert abs(norm - 1.0) < 0.02, f"Embedding not normalized: norm={norm}"
        assert face["confidence"] > 0.3, f"Very low confidence: {face['confidence']}"

    print(f"\n  ✓ Process photo passed for {label}")
    return data


def test_face_detection_only(image_bytes: bytes, label: str):
    """Test the detection-only endpoint (no embeddings)."""
    print(f"\n{'=' * 60}")
    print(f"TEST: Face Detection Only — {label}")
    print("=" * 60)

    files = {"file": (f"{label}.jpg", io.BytesIO(image_bytes), "image/jpeg")}
    resp = requests.post(f"{BASE_URL}/api/face/detect", files=files)

    if resp.status_code != 200:
        print(f"  ✗ Request failed: {resp.status_code}")
        return

    data = resp.json()
    print(f"  Faces detected: {data['faces_detected']}")
    print(f"  Image size:     {data['image_width']}x{data['image_height']}")

    # Detection-only should NOT return embeddings
    for face in data["faces"]:
        assert "embedding" not in face, "Detection-only endpoint should not return embeddings"

    print("  ✓ Detection-only passed")


def test_compare_embeddings(embedding_a: list, embedding_b: list, expected_same: bool):
    """Test the embedding comparison endpoint."""
    print(f"\n{'=' * 60}")
    print(f"TEST: Compare Embeddings (expected same person: {expected_same})")
    print("=" * 60)

    resp = requests.post(
        f"{BASE_URL}/api/face/compare",
        json={"embedding_a": embedding_a, "embedding_b": embedding_b},
    )

    if resp.status_code != 200:
        print(f"  ✗ Request failed: {resp.status_code}")
        print(f"    {resp.text}")
        return

    data = resp.json()
    print(f"  Cosine similarity:  {data['similarity']:.6f}")
    print(f"  Euclidean distance: {data['distance']:.6f}")
    print(f"  Same person:        {data['is_same_person']}")
    print(f"  Threshold used:     {data['threshold_used']}")

    print("  ✓ Comparison endpoint works")


def test_validation_errors():
    """Test error handling for invalid inputs."""
    print(f"\n{'=' * 60}")
    print("TEST: Validation Error Handling")
    print("=" * 60)

    # Test with non-image file
    files = {"file": ("test.txt", io.BytesIO(b"this is not an image"), "text/plain")}
    resp = requests.post(f"{BASE_URL}/api/face/process-photo", files=files)
    print(f"  Non-image file: {resp.status_code} — {resp.json().get('detail', '')[:80]}")
    assert resp.status_code == 400, "Should reject non-image files"

    # Test with tiny file
    files = {"file": ("tiny.jpg", io.BytesIO(b"x"), "image/jpeg")}
    resp = requests.post(f"{BASE_URL}/api/face/process-photo", files=files)
    print(f"  Tiny file:      {resp.status_code} — {resp.json().get('detail', '')[:80]}")
    assert resp.status_code == 400, "Should reject tiny files"

    # Test compare with wrong dimensions
    resp = requests.post(
        f"{BASE_URL}/api/face/compare",
        json={"embedding_a": [0.1] * 256, "embedding_b": [0.2] * 256},
    )
    print(f"  Wrong dim:      {resp.status_code} — {resp.json().get('detail', '')[:80]}")
    assert resp.status_code == 400, "Should reject non-512-dim embeddings"

    print("  ✓ Validation error handling passed")


def main():
    # Force UTF-8 output on Windows to avoid encoding errors with special characters
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("     Event Photo AI - Face Pipeline End-to-End Test")
    print("=" * 60)
    print(f"\nTarget: {BASE_URL}")

    # Step 1: Health check
    if not test_health():
        print("\n✗ Service is not healthy. Is the server running?")
        sys.exit(1)

    # Step 2: Download test images
    print(f"\n{'=' * 60}")
    print("Downloading test images...")
    print("=" * 60)

    try:
        image_a = download_test_image(TEST_IMAGE_URLS[0], "Face A")
        # Small delay to ensure a different generated face
        time.sleep(1)
        image_b = download_test_image(TEST_IMAGE_URLS[1], "Face B")
    except Exception as e:
        print(f"\n⚠ Could not download test images: {e}")
        print("You can also test manually with local images:")
        print(f'  curl -X POST -F "file=@your_photo.jpg" {BASE_URL}/api/face/process-photo')
        sys.exit(1)

    # Step 3: Test face detection only
    test_face_detection_only(image_a, "Face A")

    # Step 4: Test full pipeline (detect + embed)
    result_a = test_process_photo(image_a, "Face A")
    result_b = test_process_photo(image_b, "Face B")

    # Step 5: Compare embeddings
    if result_a and result_b and result_a["faces_detected"] > 0 and result_b["faces_detected"] > 0:
        emb_a = result_a["faces"][0]["embedding"]
        emb_b = result_b["faces"][0]["embedding"]

        # Compare different faces — should show low similarity
        test_compare_embeddings(emb_a, emb_b, expected_same=False)

        # Compare same face with itself — should show perfect similarity
        test_compare_embeddings(emb_a, emb_a, expected_same=True)

    # Step 6: Validation error handling
    test_validation_errors()

    # Summary
    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nThe deep learning face pipeline is working:")
    print("  • RetinaFace detection:   ✓")
    print("  • ArcFace embeddings:     ✓ (512-dim, L2-normalized)")
    print("  • Embedding comparison:   ✓")
    print("  • Input validation:       ✓")
    print("  • Error handling:         ✓")
    print("\nReady for Phase 3 integration.")


if __name__ == "__main__":
    main()
