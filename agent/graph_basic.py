from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, END

from agent.nodes.rewrite import rewrite_query
from agent.nodes.evaluate import evaluate
from agent.utils import format_documents
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid

# Agent State class
class AgentState(TypedDict):
    query: str
    rewritten_query: str
    results: List[Any]
    score: float
    attempts: int

#Nodes
def rewrite_node(state: AgentState):
    rewritten = rewrite_query(state["query"])
    return {**state, "rewritten_query": rewritten}


def retrieve_node(state: AgentState):
    results = retrieve_hybrid(state["rewritten_query"], top_k=10)
    return {**state, "results": results}


def evaluate_node(state: AgentState):
    docs = format_documents(state["results"])
    score = evaluate(state["query"], docs)

    print(f"Attempt {state['attempts']+1} - Score: {score}")

    return {
        **state,
        "score": score,
        "attempts": state["attempts"] + 1,
    }


THRESHOLD = 0.65
MAX_ATTEMPTS = 2

def decide_next(state: AgentState):
    if state["score"] >= THRESHOLD:
        return "end"

    if state["attempts"] >= MAX_ATTEMPTS:
        return "end"

    return "retry"

# Define the graph

graph = StateGraph(AgentState)

#nodes
graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate", evaluate_node)

graph.set_entry_point("rewrite")

#edges
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    decide_next,
    {
        "retry": "rewrite",
        "end": END,
    },
)

app = graph.compile()


# Create Runner Function
def run_agent(query: str):
    initial_state = {
        "query": query,
        "rewritten_query": "",
        "results": [],
        "score": 0.0,
        "attempts": 0,
    }

    final_state = app.invoke(initial_state)

    return final_state