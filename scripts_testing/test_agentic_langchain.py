import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # adds rag-google-io/ to path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env") # load env vars from project root

from agent.nodes.rewrite import rewrite_query
from agent.nodes.evaluate import evaluate
from agent.utils import format_documents
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid


def run_test(query: str):
    print(f"\nQuery: {query}")

    rewritten = rewrite_query(query)
    print(f"\nRewritten: {rewritten}")

    results = retrieve_hybrid(rewritten, top_k=10)

    docs = format_documents(results)
    score = evaluate(query, docs)

    print(f"\nScore: {score:.3f}")


if __name__ == "__main__":
    test_queries = [
        "What are the trends in agentic AI?",
        "What new products did Google announce at I/O related to AI?",
        "What are the Google areas of focus for AI research?",
    ]

    for q in test_queries:
        run_test(q)