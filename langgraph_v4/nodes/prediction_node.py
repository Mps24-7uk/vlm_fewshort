# nodes/prediction_node.py
import os
import numpy as np


# Global classes cache
_classes_cache = {"classes": None}


def _load_classes():
    """Lazy load and cache class labels."""
    if _classes_cache["classes"] is None:
        class_file = os.path.join(os.path.dirname(__file__), "classes.txt")
        with open(class_file) as f:
            _classes_cache["classes"] = [c.strip() for c in f.readlines()]
    return _classes_cache["classes"]


def prediction_node(state):
    classes = _load_classes()

    labels = state["neighbor_labels"]
    scores = state["similarity_scores"]

    # Use numpy operations efficiently
    unique, counts = np.unique(labels, return_counts=True)
    pred_label = int(unique[counts.argmax()])
    pred_class = classes[pred_label]
    confidence = float(np.mean(scores))

    # Store unique/counts for reuse in fp_decision_node
    state["predicted_label"] = pred_label
    state["predicted_class"] = pred_class
    state["confidence"] = round(confidence, 4)
    state["_label_counts"] = (unique, counts)  # Cache for next node

    return state
