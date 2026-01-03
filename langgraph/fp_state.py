# fp_state.py
from typing import TypedDict, List


class FPState(TypedDict):
    image_path: str

    embedding: List[float]

    similarity_scores: List[float]
    neighbor_labels: List[int]

    predicted_label: int
    predicted_class: str
    confidence: float

    fp_flag: bool
    reason: str
