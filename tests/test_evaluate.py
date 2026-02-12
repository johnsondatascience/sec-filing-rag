"""Tests for evaluation pipeline."""

import json
from pathlib import Path

from src.evaluate import load_eval_dataset, format_for_ragas


def test_load_eval_dataset(tmp_path):
    """Eval dataset loads from JSON correctly."""
    data = [
        {
            "question": "What is revenue?",
            "ground_truth": "$2B",
            "category": "factual",
            "company": "X",
        }
    ]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data))
    loaded = load_eval_dataset(path)
    assert len(loaded) == 1
    assert loaded[0]["question"] == "What is revenue?"


def test_format_for_ragas():
    """Converts our format to RAGAS-expected format."""
    eval_data = [
        {
            "question": "What is revenue?",
            "ground_truth": "$2B",
            "category": "factual",
            "company": "X",
        }
    ]
    answers = ["Revenue was $2B [1]."]
    contexts = [["Revenue grew to $2B in 2024."]]

    ragas_data = format_for_ragas(eval_data, answers, contexts)
    assert "question" in ragas_data
    assert "answer" in ragas_data
    assert "contexts" in ragas_data
    assert "ground_truth" in ragas_data
    assert len(ragas_data["question"]) == 1
