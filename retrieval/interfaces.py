from typing import Protocol, List, Optional, Dict, Any
from .result import ScoredPoint


class Retriever(Protocol):
    """
    Core retriever interface.

    Any retriever (dense, sparse, hybrid, etc.) must:
    - accept a query string
    - accept a top-k parameter
    - return a list of ScoredPoint objects
    """

    def __call__(self, query: str, k: int) -> List[ScoredPoint]:
        ...


class FilterableRetriever(Protocol):
    """
    Optional extension for retrievers that support filtering.

    Useful if you later add:
    - metadata filtering
    - namespace constraints
    - time-based filtering
    """

    def __call__(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredPoint]:
        ...


class BatchRetriever(Protocol):
    """
    Optional interface for batch retrieval.

    Useful for:
    - evaluation pipelines
    - offline benchmarking
    - throughput optimization
    """

    def __call__(
        self,
        queries: List[str],
        k: int,
    ) -> List[List[ScoredPoint]]:
        ...


class Reranker(Protocol):
    """
    Interface for reranking models (e.g., CrossEncoder).

    Takes:
    - a query
    - a list of retrieved ScoredPoints

    Returns:
    - reordered (and possibly rescored) ScoredPoints
    """

    def __call__(
        self,
        query: str,
        results: List[ScoredPoint],
    ) -> List[ScoredPoint]:
        ...


class QueryRewriter(Protocol):
    """
    Interface for query rewriting (LLM-based or rule-based).

    This is useful for:
    - LangChain-based rewriting
    - heuristic rewriting
    - future experimentation
    """

    def __call__(self, query: str) -> str:
        ...


class Evaluator(Protocol):
    """
    Interface for evaluating retrieval quality.

    Used by:
    - agent decision-making
    - evaluation pipelines

    Returns:
    - a float score (e.g., 0.0 to 1.0)
    """

    def __call__(
        self,
        query: str,
        results: List[ScoredPoint],
    ) -> float:
        ...