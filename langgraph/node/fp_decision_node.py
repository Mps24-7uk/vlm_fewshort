# nodes/fp_decision_node.py
import numpy as np


def fp_decision_node(state):
    """
    FP if:
    - Model predicts defect
    - Confidence is low OR
    - Neighbor labels are mixed
    """

    labels = state["neighbor_labels"]
    confidence = state["confidence"]
    predicted_class = state["predicted_class"]

    label_diversity = len(set(labels))

    if predicted_class != "no_defect" and (
        confidence < 0.75 or label_diversity > 2
    ):
        state["fp_flag"] = True
        state["reason"] = (
            "Low confidence or mixed neighbors for defect prediction"
        )
    else:
        state["fp_flag"] = False
        state["reason"] = "Prediction consistent with neighbors"

    return state
