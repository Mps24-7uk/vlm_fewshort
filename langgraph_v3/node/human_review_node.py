# nodes/human_review_node.py
def human_review_node(state):
    """
    Placeholder for human decision.
    In production:
    - UI button
    - QA dashboard
    - Annotation tool
    """

    # 🚨 TEMP: simulated decision
    # Replace this with real human input
    print(f"\n👤 HUMAN REVIEW REQUIRED")
    print(f"Image: {state['image_path']}")
    print(f"Prediction: {state['predicted_class']}")
    print(f"Reason: {state['reason']}")

    decision = input("Approve prediction? (y/n): ").strip().lower()

    if decision == "y":
        state["human_decision"] = "approve"
        state["fp_flag"] = False
        state["reason"] += " | Approved by human"
    else:
        state["human_decision"] = "reject"
        state["fp_flag"] = True
        state["reason"] += " | Rejected by human"

    return state
