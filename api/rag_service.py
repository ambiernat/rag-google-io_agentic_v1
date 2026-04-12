# api/services/rag_service.py

from typing import Dict, Any
from agent.graph_basic import run_agent  # or however you expose it


def run_rag(query: str) -> Dict[str, Any]:
    """
    Main entry point for RAG execution.

    This function:
    - receives a user query
    - runs the agentic retrieval pipeline
    - returns structured results
    """

    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string")

    result = run_agent(query)

    # Normalize response (important for API stability)
    return {
        "query": query,
        "answer": result.get("answer"),
        "documents": result.get("documents"),
        "score": result.get("score"),
        "strategy": result.get("strategy"),
    }