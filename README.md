# 📄 RAG Retrieval, Evaluation & Production Deployment

## Project Overview

This project implements a **production-grade Retrieval-Augmented Generation (RAG) system** for Google I/O transcripts, supporting **dense, sparse, and hybrid retrieval**, **reranking**, **offline evaluation**, and a **deployed FastAPI service on AWS**.

It combines:

- Research-grade evaluation pipelines  
- Config-driven retrieval experimentation  
- A production-ready FastAPI API  
- Containerized cloud deployment using **AWS ECS + Fargate**

---

## 🧠 Core Capabilities

### Retrieval
- **Sparse retrieval** — BM25  
- **Dense retrieval** — SentenceTransformers embeddings  
- **Hybrid retrieval** — Dense + sparse fusion  
- **Vector store** — Qdrant  

### Reranking
- CrossEncoder-based reranking
- Hyperparameter optimization (HPO)
- Offline comparison of reranking strategies

### Agentic Retrieval
- **Query rewriting** — GPT-4o-mini rewrites queries to improve retrieval vocabulary coverage
- **Self-evaluation** — LLM scores retrieved documents 0–1 for relevance against the original query
- **Retry loop** — automatic strategy switching (`hybrid → dense → sparse`) if score falls below threshold (0.65)
- **Best-result tracking** — carries forward the highest-scoring strategy result across all attempts

### Evaluation
- Recall@K, MRR, Precision@K
- Offline A/B testing
- Experiment tracking artifacts

### Production
- FastAPI search service  
- Dockerized deployment  
- AWS ECS + Fargate  
- CloudWatch logging & monitoring  

---

## 📂 Repository Structure

```text
.
├── agent/                  # Agentic retrieval layer (LangGraph)
│   ├── graph.py            # LangGraph StateGraph — rewrite → retrieve → evaluate → switch
│   ├── llm.py              # LLM factory (GPT-4o-mini via LangChain)
│   ├── utils.py            # Document formatting helpers
│   └── nodes/
│       ├── rewrite.py      # Query rewriting node
│       └── evaluate.py     # Relevance scoring node (0–1)
│
├── api/                    # FastAPI app (production)
│   ├── main.py
│   ├── routers/
│   │   ├── health.py
│   │   └── search.py
│   ├── models.py
│   └── schemas.py
│
├── retrieval/              # Retrieval logic
│   ├── retrievers/
│   │   ├── retrieve_dense.py
│   │   ├── retrieve_sparse.py
│   │   ├── retrieve_hybrid.py
│   │   └── dispatcher.py
│   └── rerankers/
│
├── vector_store/           # Qdrant ingestion
│   ├── ingest_dense.py
│   ├── ingest_sparse.py
│   └── ingest_hybrid.py
│
├── evaluation/             # Offline evaluation
│   ├── evaluate_dense.py
│   ├── evaluate_sparse.py
│   ├── evaluate_hybrid.py
│   ├── evaluate_rerank_post_hpo.py
│   └── evaluate_agentic.py
│
├── retrieval/hpo/          # HPO scripts
│   ├── hybrid_rerank_hpo.py
│   └── config_hybrid_rerank.yaml
│
├── ingestion/              # Data ingestion & preprocessing
├── data/                   # Raw, chunked, evaluation data
├── qdrant_storage/         # Local Qdrant persistence (dev)
├── tests/                  # Unit, integration & E2E tests
│
├── Dockerfile
├── docker-compose.yml
├── docker-compose_prod.yml
├── requirements.dev.txt
└── README.md
```

---

## ⚙️ Configuration

Configuration is **YAML-driven** across ingestion, retrieval, and evaluation.

**Example (`retrieval/config.yaml`):**

```yaml
qdrant:
  url: "http://localhost:6333"

collections:
  dense: "google-io-transcripts-dense"
  sparse: "google-io-transcripts-sparse"
  hybrid: "google-io-transcripts-hybrid"

retrieval:
  top_k: 5
```

---

## ▶️ Running Locally

### Docker (Recommended)

```bash
docker-compose up
```

### Example API Call

```bash
curl "http://localhost:8000/search?query=large language models&top_k=5"
```

---

## 🧪 Offline Evaluation

### Run retrieval benchmarks locally:

```bash
python evaluation/evaluate_dense.py
python evaluation/evaluate_sparse.py
python evaluation/evaluate_hybrid.py
python evaluation/evaluate_rerank_post_hpo.py
python evaluation/evaluate_agentic.py
```

### Outputs are written to:

```text
data/eval/results/
```

---

## 📊 Evaluation Results

Evaluation was conducted in five rounds, each progressively improving ground truth quality and retrieval strategy.

### Ground Truth Construction

Ground truth was generated using GPT (`gpt-4o-mini`) by sampling 2 chunks per video across 78 videos, yielding **312 queries**. Two improvements were applied iteratively:

1. **Paraphrasing** — queries were rewritten to reduce vocabulary overlap with source chunks, making evaluation more realistic and less favourable to BM25
2. **Multi-doc relevance labelling** — instead of a single relevant document per query, GPT-as-judge labelled all retrieved top-5 candidates as relevant/not relevant, giving graded rather than binary relevance

---

### Round 1 — Synthetic Ground Truth (Baseline)

Queries generated directly from source chunks. High vocabulary overlap artificially inflates BM25 performance.

| Method | Recall@5 | MRR | Precision@5 |
|--------|----------|-----|-------------|
| Dense  | 0.801 | 0.637 | 0.160 |
| Hybrid | 0.990 | 0.814 | 0.198 |
| Sparse | 0.990 | 0.943 | 0.198 |

Sparse dominates on MRR because exact keyword matching aligns perfectly with synthetically generated queries. Results are optimistic and not representative of real user queries.

---

### Round 2 — Paraphrased Ground Truth

Queries rewritten to use different vocabulary, exposing retrieval robustness more fairly.

| Method | Recall@5 | MRR | Precision@5 |
|--------|----------|-----|-------------|
| Dense  | 0.702 | 0.531 | 0.140 |
| Hybrid | 0.946 | 0.739 | 0.189 |
| Sparse | 0.917 | 0.797 | 0.183 |

Dense drops the most (-12% recall) — pure semantic search struggles when query phrasing diverges from source text. Sparse also drops, but hybrid is the most robust with the smallest decline (-4% recall), confirming that combining both signals provides resilience to query variation.

---

### Round 3 — Paraphrased + Multi-Doc Relevance (Final)

Binary relevance labels replaced with graded labels: all retrieved chunks judged relevant by GPT are counted as correct. This removes the unfair penalty for retrieving chunks that are genuinely relevant but weren't the single labelled document.

| Method | Recall@5 | MRR | Precision@5 |
|--------|----------|-----|-------------|
| Dense  | 0.974 | 0.881 | 0.490 |
| Hybrid | 0.984 | 0.922 | 0.458 |
| Sparse | 0.946 | 0.879 | 0.344 |

All methods improve substantially once genuinely relevant chunks are no longer penalised. Dense recovers strongly on precision (0.49), reflecting its ability to retrieve semantically similar content that binary labels previously marked as wrong. Hybrid remains the best overall retriever with the highest MRR (0.922) and recall (0.984). Sparse falls behind on precision, retrieving the right document at rank 1 but filling remaining slots with less relevant results.

---

### Round 4 — Hybrid + CrossEncoder Reranking

A CrossEncoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was applied on top of hybrid retrieval, reranking the top 20 candidates down to 5.

| Method | Recall@5 | MRR | Precision@5 |
|--------|----------|-----|-------------|
| Hybrid | 0.984 | 0.922 | 0.458 |
| Hybrid + CrossEncoder | 0.974 | 0.909 | 0.196 |

The reranker does not improve performance on this corpus. Precision drops significantly (0.458 → 0.196), indicating the CrossEncoder — trained on MS MARCO web search data — does not transfer well to the conversational style of Google I/O talk transcripts. Hybrid retrieval alone is the recommended production configuration.

---

### Round 5 — Agentic Retrieval (LangGraph + GPT-4o-mini)

An agentic retrieval layer was built using **LangGraph** and **LangChain**, wrapping the existing retrieval strategies in an intelligent pipeline with three components:

#### Pipeline Architecture

```
User Query
    ↓
[Rewrite Node] — GPT-4o-mini rewrites query for better retrieval coverage
    ↓
[Retrieve Node] — retrieves top-10 docs using current strategy (default: hybrid)
    ↓
[Evaluate Node] — GPT-4o-mini scores retrieved docs 0–1 against original query
    ↓
[Decision] — score ≥ 0.65 → return results | score < 0.65 → switch strategy
    ↓
[Switch Node] — hybrid → dense → sparse (fallback chain)
```

#### Key Design Decisions

- **Evaluation uses the original query** (not the rewritten one) — relevance is judged against user intent, not the rewrite
- **Best-result tracking** — the agent carries forward the highest-scoring strategy result across all attempts, returning the best rather than the last
- **LLM scoring scale** — 0.0 (irrelevant) to 1.0 (perfect match), with threshold at 0.65
- **Max attempts**: 2 per query before forcing an exit

#### Results (187 queries, paraphrased + multi-doc ground truth)

| Method | Recall@5 | MRR | Precision@5 | Avg Attempts |
|--------|----------|-----|-------------|--------------|
| Hybrid (baseline) | 0.984 | 0.922 | 0.458 | — |
| Agentic (LangGraph) | 0.9947 | 0.9149 | 0.4342 | 1.03 |

#### Strategy Breakdown

| Strategy | Queries | % |
|----------|---------|---|
| hybrid | 186 | 99.5% |
| dense | 1 | 0.5% |

**Key observations:**
- Recall improves slightly (+0.0107) over the hybrid baseline — query rewriting occasionally surfaces relevant documents that the original query missed
- MRR drops marginally (−0.0071) — the LLM evaluator's score distribution introduces occasional reordering that doesn't always align with ground truth rank
- Precision drops slightly (−0.0238) — top-10 retrieval with LLM filtering vs. top-5 direct retrieval changes the precision denominator dynamics
- The agent almost exclusively uses hybrid (99.5%) — confirming hybrid is strong enough that fallback is rarely needed
- Average of 1.03 attempts per query — the agent almost never needs to switch strategy, meaning hybrid retrieval consistently clears the 0.65 threshold

---

### Hyperparameter Optimisation (HPO)

Optuna was used to search over `retrieve_k ∈ {20, 30}`, `rerank_k ∈ {5}`, and two CrossEncoder models over 20 trials on a 50-query subsample. Results are tracked in MLflow under the `hybrid_rerank_hpo` experiment on Databricks. The best trial confirmed no meaningful gain from reranking on this dataset.

---

### ✅ Final Production Configuration

**Hybrid retrieval, top_k=5, no reranker.**

| Metric | Score |
|--------|-------|
| Recall@5 | 0.984 |
| MRR | 0.922 |
| Precision@5 | 0.458 |

The agentic layer is available as an optional enhancement for queries where retrieval confidence is low, but adds LLM latency (~5s per query) and is not recommended as the default production path on this corpus.

---

## 🤖 Agentic Layer — Implementation Details

### LangGraph State

```python
class AgentState(TypedDict):
    query: str               # original user query
    rewritten_query: str     # LLM-rewritten variant
    results: List[Any]       # current retrieval results
    score: float             # current LLM relevance score
    attempts: int            # number of strategies tried
    strategy: str            # current retrieval strategy
    best_score: float        # best score seen so far
    best_results: List[Any]  # results from best strategy
    best_strategy: str       # name of best strategy
```

### Query Rewriter

GPT-4o-mini rewrites the user query to improve retrieval keyword coverage without adding new information:

```
Rewrite the following query to improve retrieval.
Focus on clarity and keywords. Do not add extra information.
```

### Relevance Evaluator

GPT-4o-mini scores retrieved documents on a 0–1 scale:

```
0.0 = completely irrelevant
0.3 = mostly irrelevant
0.5 = partially relevant
0.7 = mostly relevant
0.9 = highly relevant
1.0 = perfect match
```

Evaluation is always performed against the **original query**, not the rewritten one.

### Running the Agent

```python
from agent.graph import run_agent

result = run_agent("What are the trends in agentic AI?")
print(result["score"])     # best relevance score
print(result["strategy"])  # winning strategy
```

---

## 🧠 Testing

- **Unit tests** — retrievers, rerankers, embeddings  
- **Integration tests** — Qdrant connectivity, collections  
- **End-to-end tests** — FastAPI search endpoint  

Run all tests with:

```bash
pytest
```

---

## 🚀 Production Deployment (AWS)

This project is fully deployed on AWS using serverless containers.

### 🐳 Docker

Production Docker image bundles FastAPI, retrieval logic, and model dependencies. **Image size**: ~550MB

### 📦 ECR — Elastic Container Registry

```text
886166401772.dkr.ecr.us-east-1.amazonaws.com/fastapi-rag:latest
```

### 🎯 ECS — Elastic Container Service

- **Cluster:** `rag-cluster`
- 2 containers: FastAPI (3 GB RAM) + Qdrant (1 GB RAM)
- Total: 1 vCPU, 4 GB RAM

### ⚡ Fargate

| State | Cost |
|-------|------|
| Running (1 task) | ~$42/month |
| Desired count = 0 | $0 |

### 🌐 Networking

```text
http://<public-ip>:8000
```

Security group: inbound TCP 8000

### 📊 CloudWatch Logs

- **Log group:** `/ecs/rag-task`
- Separate log streams per container (`fastapi`, `qdrant`)

### To Restart the App

```bash
aws ecs update-service \
  --cluster rag-cluster \
  --service rag-task3 \
  --desired-count 1 \
  --region us-east-1
```

---

## 📈 Offline & Online Experimentation

### Offline
- Metric comparison across retrieval strategies
- Reranking effectiveness
- Hyperparameter optimization with Optuna + MLflow

### Online (Foundation in Place)
The API can be extended to log queries, retrieved documents, clicks, and experiment groups — enabling production A/B testing.

---

## 🔮 Future Enhancements

- Persistent Qdrant storage via EFS
- Autoscaling ECS services
- Authentication & rate limiting
- Query analytics dashboard
- Multi-language retrieval
- Online learning from user feedback

---

## ✅ Summary

This repository implements a **complete RAG system lifecycle**:

- ✔ Research & evaluation
- ✔ Retrieval + reranking experimentation  
- ✔ Progressive ground truth improvement (synthetic → paraphrased → multi-doc)
- ✔ Agentic retrieval with LangGraph (query rewriting + self-evaluation + strategy switching)
- ✔ Production FastAPI service
- ✔ Dockerized deployment
- ✔ Serverless AWS infrastructure
