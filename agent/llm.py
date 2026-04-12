from langchain_openai import ChatOpenAI
import os


def get_llm(temperature: float = 0.0):
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature, #determinism
        max_tokens=512, #prevent overly long outputs
    )