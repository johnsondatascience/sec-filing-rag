"""Run retrieval evaluation (context quality) without LLM generation costs."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from src.chunker import Chunk
from src.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, EVAL_DIR
from src.indexer import build_bm25_index
from src.retriever import retrieve


def load_indexes():
    """Load ChromaDB collection and rebuild BM25."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION_NAME)

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


def evaluate_retrieval():
    """Evaluate retrieval quality: does the system find relevant chunks?"""
    collection, bm25, chunks = load_indexes()

    eval_path = EVAL_DIR / "questions.json"
    eval_data = json.loads(eval_path.read_text())

    results = []
    for i, item in enumerate(eval_data):
        query = item["question"]
        company = item.get("company", "")
        print(f"[{i+1}/{len(eval_data)}] {query[:60]}...")

        retrieved = retrieve(query, collection, bm25, chunks)

        # Check if any retrieved chunk mentions the target company
        companies_found = {c.company for c in retrieved}
        target_companies = set(company.split(","))
        company_hit = bool(target_companies & companies_found)

        # Check if ground truth keywords appear in retrieved text
        gt_words = set(item["ground_truth"].lower().split())
        retrieved_text = " ".join(c.text.lower() for c in retrieved)
        keyword_overlap = len(gt_words & set(retrieved_text.split())) / len(gt_words)

        results.append({
            "question": query,
            "category": item.get("category", "unknown"),
            "target_company": company,
            "company_hit": company_hit,
            "companies_found": sorted(companies_found),
            "keyword_overlap": round(keyword_overlap, 3),
            "num_retrieved": len(retrieved),
            "retrieved_sections": [
                f"{c.company} {c.year} {c.section}" for c in retrieved
            ],
        })

    # Aggregate metrics
    total = len(results)
    company_hits = sum(1 for r in results if r["company_hit"])
    avg_keyword_overlap = sum(r["keyword_overlap"] for r in results) / total

    summary = {
        "total_questions": total,
        "company_hit_rate": round(company_hits / total, 3),
        "avg_keyword_overlap": round(avg_keyword_overlap, 3),
        "by_category": {},
    }

    categories = set(r["category"] for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r["category"] == cat]
        cat_hits = sum(1 for r in cat_results if r["company_hit"])
        cat_overlap = sum(r["keyword_overlap"] for r in cat_results) / len(cat_results)
        summary["by_category"][cat] = {
            "count": len(cat_results),
            "company_hit_rate": round(cat_hits / len(cat_results), 3),
            "avg_keyword_overlap": round(cat_overlap, 3),
        }

    output = {"summary": summary, "per_question": results}

    out_path = EVAL_DIR / "retrieval_eval_results.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\n{'='*50}")
    print(f"RETRIEVAL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total questions: {total}")
    print(f"Company hit rate: {summary['company_hit_rate']:.1%}")
    print(f"Avg keyword overlap: {summary['avg_keyword_overlap']:.1%}")
    print(f"\nBy category:")
    for cat, metrics in summary["by_category"].items():
        print(f"  {cat}: hit={metrics['company_hit_rate']:.1%}, overlap={metrics['avg_keyword_overlap']:.1%} (n={metrics['count']})")

    print(f"\nResults saved to {out_path}")
    return output


if __name__ == "__main__":
    evaluate_retrieval()
