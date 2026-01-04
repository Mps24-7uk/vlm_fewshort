# fp_state.py
from typing import TypedDict, List


class FPState(TypedDict):
    # Input
    image_path: str

    # Model outputs
    embedding: List[float]
    similarity_scores: List[float]
    neighbor_labels: List[int]

    # Prediction
    predicted_label: int
    predicted_class: str
    confidence: float

    # FP decision
    fp_flag: bool
    fp_score: float
    reason: str
