# nodes/fp_decision_node.py
from fp_config import FP_CONFIG

def fp_decision_node(state):
    confidence = state["confidence"]
    predicted_class = state["predicted_class"]
    
    # Reuse cached label counts from prediction_node
    unique, counts = state.get("_label_counts", ([], []))
    
    if not len(counts):
        # Fallback if cache not available
        import numpy as np
        labels = state["neighbor_labels"]
        unique, counts = np.unique(labels, return_counts=True)
    
    majority_ratio = counts.max() / len(state["neighbor_labels"])

    CONFIDENCE_THRESHOLD = FP_CONFIG["confidence_threshold"]
    MAJORITY_THRESHOLD = FP_CONFIG["majority_threshold"]

    if confidence < CONFIDENCE_THRESHOLD or majority_ratio < MAJORITY_THRESHOLD:
        state["fp_flag"] = True
        state["requires_human_review"] = True
        state["reason"] = (
            f"Unstable prediction for '{predicted_class}' "
            f"(confidence={confidence}, majority={majority_ratio:.2f})"
        )
    else:
        state["fp_flag"] = False
        state["requires_human_review"] = False
        state["reason"] = f"Stable prediction for '{predicted_class}'"

    return state
