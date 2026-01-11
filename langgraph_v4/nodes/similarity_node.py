# nodes/similarity_node.py
import os
import faiss
import numpy as np
from fp_config import FP_CONFIG


INDEX_PATH = os.path.join(os.path.dirname(__file__), "chip_resnet.index")

# Global index cache
_index_cache = {"index": None}


def _get_index():
    """Lazy load and cache the FAISS index."""
    if _index_cache["index"] is None:
        _index_cache["index"] = faiss.read_index(INDEX_PATH)
    return _index_cache["index"]


def similarity_node(state):
    index = _get_index()

    emb = np.array(state["embedding"], dtype="float32").reshape(1, -1)
    scores, ids = index.search(emb, FP_CONFIG["top_k"])

    state["similarity_scores"] = scores[0].tolist()
    state["neighbor_labels"] = ids[0].tolist()
    return state
