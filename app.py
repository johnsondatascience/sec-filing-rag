"""Streamlit UI for SEC Filing RAG with citations."""

import streamlit as st
import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, TICKERS
from src.chunker import Chunk
from src.generator import generate_answer
from src.indexer import build_bm25_index
from src.retriever import retrieve


st.set_page_config(page_title="SEC Filing RAG", page_icon="📊", layout="wide")
st.title("📊 SEC Filing RAG with Citations")
st.caption("Ask questions about public company 10-K filings — every answer is grounded and cited.")


@st.cache_resource
def load_indexes():
    """Load ChromaDB collection and rebuild BM25 from stored data."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION_NAME)

    # Retrieve all documents to rebuild BM25 and chunk lookup
    all_data = collection.get(include=["documents", "metadatas"])
    chunks = []
    for doc_id, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        chunks.append(
            Chunk(
                text=doc,
                company=meta["company"],
                year=meta["year"],
                section=meta["section"],
                chunk_id=doc_id,
                strategy=meta.get("strategy", "unknown"),
            )
        )

    bm25, aligned = build_bm25_index(chunks)
    return collection, bm25, aligned


# Sidebar
with st.sidebar:
    st.header("Filters")
    selected_companies = st.multiselect(
        "Companies",
        options=TICKERS,
        default=TICKERS,
        help="Filter retrieval to specific companies",
    )
    st.divider()
    st.header("About")
    st.markdown(
        "This RAG system uses **hybrid search** (BM25 + dense embeddings) "
        "with **reciprocal rank fusion** and a **cross-encoder reranker** "
        "to find relevant passages from SEC 10-K filings."
    )

# Load indexes
try:
    collection, bm25, chunks = load_indexes()
    st.sidebar.success(f"Loaded {collection.count()} indexed chunks")
except Exception as e:
    st.error(f"Failed to load indexes. Run the ingestion pipeline first.\n\n{e}")
    st.stop()

# Query input
query = st.text_input(
    "Ask a question about SEC filings:",
    placeholder="What were Netflix's main risk factors in their most recent 10-K?",
)

if query:
    with st.spinner("Searching and generating answer..."):
        # Retrieve
        result_chunks = retrieve(query, collection, bm25, chunks)

        # Generate
        result = generate_answer(query, result_chunks)

    # Display answer
    st.subheader("Answer")
    st.markdown(result["answer"])

    # Display sources
    if result["sources"]:
        st.subheader("Sources")
        for num, source in sorted(result["sources"].items()):
            with st.expander(source["label"]):
                st.markdown(f"**{source['company']}** — {source['section']} ({source['year']})")
                st.text(source["text"])

    # Display all retrieved chunks for transparency
    with st.expander("All retrieved chunks (for transparency)"):
        for i, chunk in enumerate(result_chunks):
            st.markdown(f"**Chunk {i+1}** — {chunk.company} {chunk.year}, {chunk.section}")
            st.text(chunk.text[:500])
            st.divider()
