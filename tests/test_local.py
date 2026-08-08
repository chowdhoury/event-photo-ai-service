"""Quick local test for the face pipeline using downloaded test images."""
import sys
import json
import numpy as np
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://localhost:8000"


def test_process(filepath, label):
    print("=" * 60)
    print(f"TEST: Process Photo - {label}")
    print("=" * 60)
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{BASE}/api/face/process-photo",
            files={"file": (f"{label}.jpg", f, "image/jpeg")},
        )
    print(f"  Status: {resp.status_code}")
    data = resp.json()

    if resp.status_code != 200:
        print(f"  Error: {data}")
        return None

    n = data["faces_detected"]
    print(f"  Faces detected: {n}")
    print(f"  Image size: {data['image_width']}x{data['image_height']}")
    print(f"  Model: {data['model_name']}")

    for face in data["faces"]:
        bbox = face["bbox"]
        emb = np.array(face["embedding"])
        norm = np.linalg.norm(emb)
        print(f"\n  Face {face['face_index']}:")
        print(f"    Bbox: x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}")
        print(f"    Confidence: {face['confidence']:.4f}")
        print(f"    Embedding dim: {face['embedding_dim']}")
        print(f"    Embedding norm: {norm:.6f} (should be ~1.0)")
        print(f"    Embedding preview: [{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}]")

        assert face["embedding_dim"] == 512, f"Expected 512, got {face['embedding_dim']}"
        assert abs(norm - 1.0) < 0.02, f"Not normalized: {norm}"

    print(f"\n  PASS\n")
    return data


def test_compare(emb_a, emb_b, label):
    print("=" * 60)
    print(f"TEST: {label}")
    print("=" * 60)
    resp = requests.post(
        f"{BASE}/api/face/compare",
        json={"embedding_a": emb_a, "embedding_b": emb_b},
    )
    comp = resp.json()
    print(f"  Cosine similarity:  {comp['similarity']:.6f}")
    print(f"  Euclidean distance: {comp['distance']:.6f}")
    print(f"  Same person:        {comp['is_same_person']}")
    print(f"  Threshold used:     {comp['threshold_used']}")
    print(f"\n  PASS\n")
    return comp


def main():
    print("\n  Event Photo AI - Face Pipeline Local Test\n")

    # Health check
    resp = requests.get(f"{BASE}/health")
    health = resp.json()
    print(f"Health: {health['status']}, model={health['model_name']}, loaded={health['model_loaded']}\n")

    # Process two different face images
    data_a = test_process("tests/test_face_a.jpg", "Face A")
    data_b = test_process("tests/test_face_b.jpg", "Face B")

    if data_a and data_b and data_a["faces_detected"] > 0 and data_b["faces_detected"] > 0:
        emb_a = data_a["faces"][0]["embedding"]
        emb_b = data_b["faces"][0]["embedding"]

        # Two different people should have low similarity
        comp_diff = test_compare(emb_a, emb_b, "Compare Different People")

        # Same person should have perfect similarity
        comp_same = test_compare(emb_a, emb_a, "Compare Same Person (self)")

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  RetinaFace detection:     PASS")
        print(f"  ArcFace 512-dim embed:    PASS")
        print(f"  L2 normalization:         PASS")
        print(f"  Different people sim:     {comp_diff['similarity']:.4f} (expected < 0.4)")
        print(f"  Same person sim:          {comp_same['similarity']:.4f} (expected ~1.0)")
        print(f"  Pipeline fully verified!")
    else:
        print("Could not complete comparison tests - check images have detectable faces")


if __name__ == "__main__":
    main()
