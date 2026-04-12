from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, END
from agent.nodes.rewrite import rewrite_query

from agent.nodes.evaluate import evaluate
from agent.utils import format_documents

from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid
from retrieval.retrievers.retrieve_dense import retrieve_dense
from retrieval.retrievers.retrieve_sparse import retrieve_sparse

class AgentState(TypedDict):
    query: str
    rewritten_query: str
    results: List[Any]
    score: float
    attempts: int
    strategy: str #Phase 2: add strategy to state

    # Phase 3: track best results across strategies
    best_score: float
    best_results: List[Any]
    best_strategy: str

STRATEGIES = ["hybrid", "dense", "sparse"]
THRESHOLD = 0.65
MAX_ATTEMPTS = 2

def rewrite_node(state: AgentState):
    rewritten = rewrite_query(state["query"])
    return {**state, "rewritten_query": rewritten}

def retrieve_node(state: AgentState):
    strategy = state["strategy"]
    if strategy == "hybrid":
        results = retrieve_hybrid(state["rewritten_query"], top_k=10)
    elif strategy == "dense":
        results = retrieve_dense(state["rewritten_query"], top_k=10)
    elif strategy == "sparse":
        results = retrieve_sparse(state["rewritten_query"], top_k=10)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    print(f"[Retrieval] Strategy: {strategy}")
    return {**state, "results": results}

# def evaluate_node(state: AgentState):
#     docs = format_documents(state["results"])
#     score = evaluate(state["rewritten_query"], docs)
#     #logging results
#     print(f"[Evaluate] Strategy: {state['strategy']} | Score: {score:.3f}")  # ← print strategy and score
#     return {**state, "score": score, "attempts": state["attempts"] + 1}

def evaluate_node(state: AgentState):
    docs = format_documents(state["results"])
    score = evaluate(state["query"], docs)

    print(f"[ScoreDist] {score:.3f}") # evaluate agentic score distribution across strategies
    print(f"[Evaluate] Strategy: {state['strategy']} | Score: {score:.3f}")

    # update best result
    if score > state["best_score"]:
        print(f"[Best] New best strategy: {state['strategy']} ({score:.3f})")

        best_score = score
        best_results = state["results"]
        best_strategy = state["strategy"]
    else:
        best_score = state["best_score"]
        best_results = state["best_results"]
        best_strategy = state["best_strategy"]

    return {
        **state,
        "score": score,
        "attempts": state["attempts"] + 1,

        # Phase 3: carry forward best results
        "best_score": best_score,
        "best_results": best_results,
        "best_strategy": best_strategy,
    }


def decide_next(state: AgentState):
    if state["score"] >= THRESHOLD:
        return "end"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "end"
    current_idx = STRATEGIES.index(state["strategy"])

    print(f"[Decision] score={state['score']:.3f} | "
    f"threshold={THRESHOLD} | "
    f"attempts={state['attempts']}/{MAX_ATTEMPTS}")

    if current_idx + 1 < len(STRATEGIES):
        return "switch"
    return "end"

def switch_strategy_node(state: AgentState):
    current_idx = STRATEGIES.index(state["strategy"])
    next_strategy = STRATEGIES[current_idx + 1]
    print(f"[Switch] {state['strategy']} → {next_strategy}")
    return {**state, "strategy": next_strategy}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("switch", switch_strategy_node)

graph.set_entry_point("rewrite")
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "evaluate")
graph.add_conditional_edges(
    "evaluate",
    decide_next,
    {
        "switch": "switch",
        "end": END,
    },
)
graph.add_edge("switch", "retrieve")

app = graph.compile()

# def run_agent(query: str) -> dict:
#     initial_state = {
#         "query": query,
#         "rewritten_query": "",
#         "results": [],
#         "score": 0.0,
#         "attempts": 0,
#         "strategy": "hybrid",  # start with hybrid strategy

#     }
#     return app.invoke(initial_state)

def run_agent(query: str, threshold: float = THRESHOLD, max_attempts: int = MAX_ATTEMPTS) -> dict:
    initial_state = {
        "query": query,
        "rewritten_query": "",
        "results": [],
        "score": 0.0,
        "attempts": 0,
        "strategy": "hybrid",  # start with hybrid strategy

        # Phase 3: initialize best results
        "best_score": 0.0,
        "best_results": [],
        "best_strategy": "hybrid",

    }

    final_state = app.invoke(initial_state)

    # return best result instead of last
    final_state["results"] = final_state["best_results"]
    final_state["score"] = final_state["best_score"]
    final_state["strategy"] = final_state["best_strategy"]

    return final_state