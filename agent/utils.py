def format_documents(results):
    return "\n\n".join([
        f"[Score: {r.score:.3f}] {r.payload.get('text', '')}"
        for r in results[:5]  # top-k only
    ])