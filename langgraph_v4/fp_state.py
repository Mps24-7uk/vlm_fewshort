
# fp_state.py
from typing import TypedDict, List, Optional


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

    # 👤 Human-in-loop
    requires_human_review: bool
    human_decision: Optional[str]   # "approve" | "reject" | None
