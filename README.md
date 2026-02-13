# SEC Filing RAG with Citations

Production-grade Retrieval-Augmented Generation system for querying SEC 10-K filings. Every answer is grounded in source documents with inline citations.

## Architecture

```
Query → Hybrid Search (BM25 + Dense) → RRF Fusion → Cross-Encoder Reranking → LLM Generation w/ Citations
```

**Retrieval Pipeline:**
- **Dense search:** `BAAI/bge-large-en-v1.5` embeddings (1024-dim) stored in ChromaDB
- **Sparse search:** BM25Okapi over tokenized chunks
- **Fusion:** Reciprocal Rank Fusion (k=60) combining both rankings
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-12-v2` for final relevance scoring

**Generation:**
- Claude API with citation-enforcing system prompt
- Source references mapped back to original filing sections
- Inline `[1]`, `[2]` citations linked to expandable source cards

## Data

8 public companies, 2 most recent 10-K filings each:

| Ticker | Company | Industry |
|--------|---------|----------|
| PYPL | PayPal | Fintech |
| XYZ | Block (Square) | Fintech |
| NFLX | Netflix | Streaming |
| DIS | Disney | Entertainment |
| CMG | Chipotle | Restaurant |
| TGT | Target | Retail |
| NET | Cloudflare | Cloud/Security |
| DDOG | Datadog | Observability |

**Corpus:** 2,155 chunks indexed from 16 filings across 7 sections (Items 1, 1A, 7, 7A, 8).

## Evaluation Results

Retrieval quality evaluated against 25 ground-truth Q&A pairs:

| Metric | Score |
|--------|-------|
| Company Hit Rate | 100% |
| Avg Keyword Overlap | 70.9% |

**By question category:**

| Category | Hit Rate | Keyword Overlap | Count |
|----------|----------|-----------------|-------|
| Analytical | 100% | 74.1% | 11 |
| Factual | 100% | 71.7% | 10 |
| Summary | 100% | 60.9% | 3 |
| Comparison | 100% | 57.1% | 1 |

## Project Structure

```
sec-filing-rag/
├── app.py                  # Streamlit UI
├── src/
│   ├── config.py           # Constants and paths
│   ├── downloader.py       # SEC EDGAR 10-K downloader
│   ├── parser.py           # HTML → structured sections
│   ├── chunker.py          # Fixed + section-aware chunking
│   ├── embedder.py         # BGE-large embedding (GPU)
│   ├── indexer.py          # ChromaDB + BM25 indexing
│   ├── retriever.py        # Hybrid search + RRF + reranking
│   ├── generator.py        # Claude API with citations
│   ├── pipeline.py         # End-to-end ingestion
│   └── evaluate.py         # RAGAS evaluation framework
├── scripts/
│   └── run_eval.py         # Retrieval evaluation script
├── tests/                  # 20 unit tests
├── data/
│   ├── raw/                # Downloaded EDGAR filings
│   ├── processed/          # Chunk metadata
│   └── eval/               # Ground-truth Q&A + results
└── chroma_db/              # Persistent vector store
```

## Setup

```bash
# Clone and install
git clone <repo-url> && cd sec-filing-rag
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# For GPU acceleration (recommended)
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124

# Set environment variables
echo "ANTHROPIC_API_KEY=your-key" > .env
echo "HF_TOKEN=your-token" >> .env
```

## Usage

### 1. Ingest filings

```bash
python -m src.pipeline
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Run evaluation

```bash
python -m scripts.run_eval
```

### 4. Run tests

```bash
pytest
```

## Key Design Decisions

- **Hybrid search over dense-only:** BM25 catches exact keyword matches (ticker symbols, financial terms) that dense embeddings can miss
- **RRF over learned fusion:** Parameter-free, robust to score distribution differences between BM25 and cosine similarity
- **Cross-encoder reranking:** Handles the precision/recall tradeoff — retrieve broadly (top 10), then rerank precisely (top 5)
- **Local embeddings:** `bge-large-en-v1.5` runs on-device for privacy and cost control; only generation uses the Claude API
- **Citation enforcement:** System prompt requires `[N]` citations; post-processing maps them back to source chunks

## Tech Stack

- **Python 3.13** with type hints
- **ChromaDB** — persistent vector store
- **sentence-transformers** — local embeddings and reranking
- **rank-bm25** — sparse retrieval
- **Anthropic Claude API** — generation with citations
- **Streamlit** — interactive demo UI
- **RAGAS** — evaluation framework
- **PyTorch + CUDA** — GPU-accelerated embedding
