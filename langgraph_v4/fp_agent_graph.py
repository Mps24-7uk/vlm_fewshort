# fp_agent_graph.py
from langgraph.graph import StateGraph, END

from fp_state import FPState
from nodes.embedding_node import embedding_node
from nodes.similarity_node import similarity_node
from nodes.prediction_node import prediction_node
from nodes.fp_decision_node import fp_decision_node
from nodes.human_review_node import human_review_node


def route_after_fp(state):
    """Route based on human review requirement."""
    return "human_review" if state["requires_human_review"] else END


# Build and cache the graph once
_graph_cache = None


def build_fp_agent():
    """Build or return cached LangGraph agent."""
    global _graph_cache
    
    if _graph_cache is not None:
        return _graph_cache
    
    graph = StateGraph(FPState)

    graph.add_node("embedding", embedding_node)
    graph.add_node("similarity", similarity_node)
    graph.add_node("prediction", prediction_node)
    graph.add_node("fp_decision", fp_decision_node)
    graph.add_node("human_review", human_review_node)

    graph.set_entry_point("embedding")

    # Linear pipeline edges
    graph.add_edge("embedding", "similarity")
    graph.add_edge("similarity", "prediction")
    graph.add_edge("prediction", "fp_decision")

    # Conditional edge for human review
    graph.add_conditional_edges(
        "fp_decision",
        route_after_fp,
        {
            "human_review": "human_review",
            END: END
        }
    )

    graph.add_edge("human_review", END)

    _graph_cache = graph.compile()
    return _graph_cache
