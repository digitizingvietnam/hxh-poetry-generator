"""
main.py

Flask API server for the Retrieval-Augmented Generation (RAG) system.

This module exposes HTTP endpoints that allow clients to:
- Submit natural language queries to the RAG pipeline
- Retrieve generated responses based on vector search
- Perform health checks for deployment monitoring
- Fetch sample historical text for frontend display or animation

The API delegates all retrieval and generation logic to the
`query_rag` utility function to ensure separation of concerns
between application logic and RAG implementation.
"""

from flask import Flask, jsonify, request, render_template, session
from utils.rag import query_rag
import os


app = Flask(__name__)
app.secret_key = "123456"  # Change this to a secure secret key in production


@app.route("/")
def index():
    """Render the main page and clear session."""
    session.clear()
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    topic = data.get("topic")
    num_lines = data.get("num_lines", 8)

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    try:
        # Call the RAG function
        result = query_rag(topic=topic, num_lines=num_lines)
        return jsonify(result)
    except Exception as e:
        print(f"Error generating poem: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Check server health for deployment monitoring."""
    return jsonify(
        {
            "status": "healthy",
            "message": "Server is running fine!",
        }
    ), 200


# Deploy configuration
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5555))
    app.run(host="0.0.0.0", port=port)
