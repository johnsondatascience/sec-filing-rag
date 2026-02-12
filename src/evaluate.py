"""RAGAS evaluation pipeline for RAG quality metrics."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics.collections import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import EVAL_DIR


def load_eval_dataset(path: Path | None = None) -> list[dict]:
    """Load ground-truth Q&A pairs from JSON."""
    path = path or (EVAL_DIR / "questions.json")
    return json.loads(path.read_text())


def format_for_ragas(
    eval_data: list[dict],
    answers: list[str],
    contexts: list[list[str]],
) -> dict[str, list]:
    """Convert to RAGAS dataset format."""
    return {
        "question": [d["question"] for d in eval_data],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [d["ground_truth"] for d in eval_data],
    }


def run_evaluation(
    eval_data: list[dict],
    answers: list[str],
    contexts: list[list[str]],
) -> dict:
    """Run RAGAS evaluation and return metric scores.

    Returns dict with per-question scores and aggregate metrics.
    """
    ragas_data = format_for_ragas(eval_data, answers, contexts)
    dataset = Dataset.from_dict(ragas_data)

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return {
        "aggregate": {k: float(v) for k, v in result.items() if isinstance(v, (int, float))},
        "per_question": result.to_pandas().to_dict(orient="records") if hasattr(result, "to_pandas") else [],
    }


def run_pipeline_evaluation(
    retrieve_fn,
    generate_fn,
    eval_path: Path | None = None,
) -> dict:
    """End-to-end evaluation: load questions → retrieve → generate → score.

    retrieve_fn: query -> list[Chunk]
    generate_fn: (query, chunks) -> dict with 'answer' key
    """
    eval_data = load_eval_dataset(eval_path)
    answers = []
    contexts = []

    for item in eval_data:
        query = item["question"]
        chunks = retrieve_fn(query)
        result = generate_fn(query, chunks)
        answers.append(result["answer"])
        contexts.append([c.text for c in chunks])

    return run_evaluation(eval_data, answers, contexts)
