from langchain_core.prompts import ChatPromptTemplate
from agent.llm import get_llm

llm = get_llm()

prompt = ChatPromptTemplate.from_template(
    """Rewrite the following query to improve retrieval.
Focus on clarity and keywords. Do not add extra information.

Query:
{query}
"""
)

def rewrite_query(query: str) -> str:
    response = (prompt | llm).invoke({"query": query})
    return response.content.strip()