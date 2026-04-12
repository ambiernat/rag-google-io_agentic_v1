from langchain_core.prompts import ChatPromptTemplate
from agent.llm import get_llm

llm = get_llm()

prompt = ChatPromptTemplate.from_template(
    """You are evaluating retrieval quality.

Query:
{query}

Retrieved Documents:
{documents}

Score relevance from 0 to 1 using this scale:

0.0 = completely irrelevant
0.3 = mostly irrelevant, weak connection
0.5 = partially relevant, missing key information
0.7 = mostly relevant, useful but not perfect
0.9 = highly relevant, directly answers the query
1.0 = perfect match

Return ONLY a number between 0 and 1.

"""
)

def evaluate(query: str, documents: str) -> float:
    response = (prompt | llm).invoke({
        "query": query,
        "documents": documents,
    })

    text = response.content.strip()

    try:
        return float(text)
    except ValueError:
        # fallback (important)
        import re
        match = re.search(r"\d*\.?\d+", text)
        return float(match.group()) if match else 0.0