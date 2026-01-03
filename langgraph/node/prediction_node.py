# nodes/prediction_node.py
import numpy as np


CLASSES_PATH = "classes.txt"


def load_classes():
    with open(CLASSES_PATH) as f:
        return [line.strip() for line in f.readlines()]


def prediction_node(state):
    classes = load_classes()

    labels = state["neighbor_labels"]
    scores = state["similarity_scores"]

    pred_label = int(np.bincount(labels).argmax())
    pred_class = classes[pred_label]
    confidence = float(np.mean(scores))

    state["predicted_label"] = pred_label
    state["predicted_class"] = pred_class
    state["confidence"] = round(confidence, 4)

    return state
