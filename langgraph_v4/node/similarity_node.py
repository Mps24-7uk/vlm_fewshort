# nodes/similarity_node.py
import faiss
import numpy as np
from fp_config import FP_CONFIG


INDEX_PATH = "chip_resnet.index"


def similarity_node(state):
    index = faiss.read_index(INDEX_PATH)

    emb = np.array(state["embedding"], dtype="float32").reshape(1, -1)
    scores, ids = index.search(emb, FP_CONFIG["top_k"])

    state["similarity_scores"] = scores[0].tolist()
    state["neighbor_labels"] = ids[0].tolist()
    return state
