import numpy as np


def fp_decision_node(state):
    """
    Simple, class-agnostic FP logic.

    FP if:
    - Confidence is low OR
    - Top-K neighbors do not strongly agree
    """

    labels = state["neighbor_labels"]
    confidence = state["confidence"]
    predicted_class = state["predicted_class"]

    # ---- Basic stats ----
    unique, counts = np.unique(labels, return_counts=True)
    majority_ratio = counts.max() / len(labels)

    # ---- Tunable thresholds ----
    CONFIDENCE_THRESHOLD = 0.75
    MAJORITY_THRESHOLD = 0.6

    # ---- FP decision ----
    if confidence < CONFIDENCE_THRESHOLD or majority_ratio < MAJORITY_THRESHOLD:
        state["fp_flag"] = True
        state["reason"] = (
            f"Unstable prediction for '{predicted_class}' "
            f"(confidence={confidence}, majority={majority_ratio:.2f})"
        )
    else:
        state["fp_flag"] = False
        state["reason"] = (
            f"Stable prediction for '{predicted_class}'"
        )

    return state
