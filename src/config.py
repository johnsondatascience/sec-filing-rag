"""Project configuration — paths, model names, constants."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# SEC EDGAR
SEC_COMPANY_NAME = "JobSearchPortfolio"
SEC_EMAIL = "michael@johnsondatascience.com"
TICKERS = ["PYPL", "XYZ", "NFLX", "DIS", "CMG", "TGT", "NET", "DDOG", "LZ"]

# Models (all run locally on RTX A4500)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Chunking
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens

# Retrieval
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5

# ChromaDB
CHROMA_COLLECTION_NAME = "sec_10k_filings"

# Generation
MAX_CONTEXT_CHUNKS = 5
