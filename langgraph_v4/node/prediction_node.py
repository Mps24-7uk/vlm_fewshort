# nodes/prediction_node.py
import numpy as np


def _load_classes():
    with open("classes.txt") as f:
        return [c.strip() for c in f.readlines()]


def prediction_node(state):
    classes = _load_classes()

    labels = state["neighbor_labels"]
    scores = state["similarity_scores"]

    unique, counts = np.unique(labels, return_counts=True)
    pred_label = int(unique[counts.argmax()])
    pred_class = classes[pred_label]
    confidence = float(np.mean(scores))

    state["predicted_label"] = pred_label
    state["predicted_class"] = pred_class
    state["confidence"] = round(confidence, 4)

    return state
