# nodes/human_review_node.py
from .ui import start_review_ui


def human_review_node(state):
    """
    Human review node with a simple FastAPI UI.
    Displays the image and two buttons: Approve and Reject.
    Sets decision value to 'y' or 'n' based on button click.
    """
    # Start the UI and get the decision
    decision = start_review_ui(state)
    if decision == "y":
        state["human_decision"] = "approve"
        state["fp_flag"] = False
        state["reason"] += " | Approved by human"
    else:
        state["human_decision"] = "reject"
        state["fp_flag"] = True
        state["reason"] += " | Rejected by human"

    return state
