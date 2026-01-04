# nodes/fp_decision_node.py
import numpy as np
from fp_config import FP_CONFIG


def fp_decision_node(state):
    labels = state["neighbor_labels"]
    confidence = state["confidence"]
    predicted_class = state["predicted_class"]

    unique, counts = np.unique(labels, return_counts=True)

    total_k = len(labels)
    majority_ratio = counts.max() / total_k
    diversity_ratio = len(unique) / total_k

    fp_score = 0.0
    reasons = []

    if confidence < FP_CONFIG["confidence_threshold"]:
        fp_score += 0.4
        reasons.append("low confidence")

    if majority_ratio < FP_CONFIG["majority_ratio_threshold"]:
        fp_score += 0.3
        reasons.append("weak neighbor majority")

    if diversity_ratio > FP_CONFIG["diversity_ratio_threshold"]:
        fp_score += 0.3
        reasons.append("high class diversity")

    state["fp_flag"] = fp_score >= 0.5
    state["fp_score"] = round(fp_score, 2)

    state["reason"] = (
        f"Unstable prediction for '{predicted_class}': "
        + ", ".join(reasons)
        if state["fp_flag"]
        else f"Stable prediction for '{predicted_class}'"
    )

    return state
