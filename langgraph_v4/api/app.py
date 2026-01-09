"""
Backend API for Human Review System
Integrates with the LangGraph FP detection pipeline
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# In-memory storage for demo (replace with database in production)
review_queue = []
review_history = {'approved': 0, 'rejected': 0}

# TODO: Import your FP agent and state management
# from fp_agent_graph import create_fp_agent
# from fp_state import FPState


@app.route('/api/review/items', methods=['GET'])
def get_review_items():
    """
    Fetch pending review items
    Returns items that require human verification
    """
    try:
        # TODO: Fetch actual items from your pipeline
        # This should query items with requires_human_review=True
        
        items = [
            {
                "id": i + 1,
                "image_path": f"/sample_images/image_{i+1}.jpg",
                "predicted_class": "Class A",
                "predicted_label": 0,
                "confidence": 0.65 + (i * 0.1),
                "reason": f"Sample prediction requiring human review {i+1}",
                "neighbor_labels": [0, 1, 2, 0, 1],
                "timestamp": datetime.now().isoformat()
            }
            for i in range(min(4, len(review_queue)))
        ]
        
        stats = {
            "pending": len(items),
            "approved": review_history['approved'],
            "rejected": review_history['rejected']
        }
        
        return jsonify({
            "items": items,
            "stats": stats,
            "success": True
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/api/review/<int:item_id>/decision', methods=['POST'])
def submit_review_decision(item_id):
    """
    Submit human review decision for an item
    
    Expected JSON:
    {
        "decision": "approve" | "reject"
    }
    """
    try:
        data = request.get_json()
        decision = data.get('decision')
        
        if decision not in ['approve', 'reject']:
            return jsonify({
                "error": "Invalid decision. Must be 'approve' or 'reject'",
                "success": False
            }), 400
        
        # TODO: Update your FP agent state with human decision
        # This should:
        # 1. Find the item in your pipeline
        # 2. Update: human_decision, fp_flag, reason
        # 3. Continue the pipeline execution
        
        # Demo: Update statistics
        if decision == 'approve':
            review_history['approved'] += 1
        else:
            review_history['rejected'] += 1
        
        # Remove from queue (demo)
        review_queue[:] = [item for item in review_queue if item.get('id') != item_id]
        
        return jsonify({
            "success": True,
            "decision": decision,
            "item_id": item_id,
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "pending": len(review_queue),
                "approved": review_history['approved'],
                "rejected": review_history['rejected']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/api/review/stats', methods=['GET'])
def get_review_stats():
    """Get overall review statistics"""
    try:
        stats = {
            "pending": len(review_queue),
            "approved": review_history['approved'],
            "rejected": review_history['rejected'],
            "total": len(review_queue) + review_history['approved'] + review_history['rejected']
        }
        
        if stats['total'] > 0:
            stats['approval_rate'] = (stats['approved'] / stats['total']) * 100
        else:
            stats['approval_rate'] = 0
            
        return jsonify({
            "stats": stats,
            "success": True
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "FP Human Review API",
        "version": "1.0.0",
        "endpoints": {
            "GET /api/review/items": "Get pending review items",
            "POST /api/review/<id>/decision": "Submit review decision",
            "GET /api/review/stats": "Get review statistics",
            "GET /api/health": "Health check"
        }
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
