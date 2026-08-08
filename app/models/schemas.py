from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int = Field(description="Top-left x coordinate")
    y: int = Field(description="Top-left y coordinate")
    width: int = Field(description="Bounding box width")
    height: int = Field(description="Bounding box height")


class FaceDetectionResult(BaseModel):
    face_index: int = Field(description="Index of the face in the image")
    bbox: BoundingBox
    confidence: float = Field(description="Detection confidence score")
    landmarks: list[list[float]] = Field(
        description="5 facial landmark points: left eye, right eye, nose, left mouth, right mouth"
    )


class FaceEmbeddingResult(BaseModel):
    face_index: int
    bbox: BoundingBox
    confidence: float
    embedding: list[float] = Field(description="512-dimensional ArcFace embedding vector")
    embedding_dim: int = Field(description="Dimensionality of the embedding vector")
    landmarks: list[list[float]]


class ImageProcessingResponse(BaseModel):
    success: bool
    faces_detected: int = Field(description="Total number of faces found in the image")
    image_width: int
    image_height: int
    model_name: str = Field(description="Name of the InsightFace model pack used")
    faces: list[FaceEmbeddingResult]


class FaceComparisonRequest(BaseModel):
    embedding_a: list[float] = Field(description="First face embedding vector")
    embedding_b: list[float] = Field(description="Second face embedding vector")


class FaceComparisonResponse(BaseModel):
    similarity: float = Field(description="Cosine similarity between the two embeddings")
    distance: float = Field(description="Euclidean distance between the two embeddings")
    is_same_person: bool = Field(
        description="Whether the embeddings likely belong to the same person (using default threshold)"
    )
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    version: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str | None = None
