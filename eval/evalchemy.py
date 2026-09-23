"""Benchmark questions and answer grading.

Grading is unchanged from the runs behind the paper: the prediction is the last \\boxed{...} in the
response (`last_boxed_only_string` + `remove_boxed`) and is compared with the gold answer using
lm_eval's `is_equiv` (Hendrycks MATH normalisation).
"""

from __future__ import annotations

import json
from pathlib import Path

from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

DATA_DIR = Path(__file__).resolve().parent / "evalchemy_data"


def _read_jsonl(name: str) -> list[dict]:
    with open(DATA_DIR / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_arrow(name: str) -> list[dict]:
    from datasets import load_from_disk

    return [dict(ex) for ex in load_from_disk(str(DATA_DIR / name))["train"]]


# name -> (loader, question key, gold-answer key)
DATASETS = {
    "MATH-500": (lambda: _read_jsonl("math500.jsonl"), "problem", "answer"),
    "AIME-2024": (lambda: _read_jsonl("aime24.json"), "problem", "expected_answer"),
    "AIME-2025": (lambda: _read_jsonl("aime25.json"), "problem", "answer"),
    "AIME-2026": (lambda: _read_arrow("aime2026"), "problem", "answer"),
    "AMC-23": (lambda: _read_jsonl("amc23.json"), "question", "answer"),
    "HMMT-2026": (lambda: _read_jsonl("hmmt2026.jsonl"), "problem", "answer"),
}
DATASET_CHOICES = tuple(DATASETS)


class Benchmark:
    def __init__(self, name: str):
        loader, question_key, self.answer_key = DATASETS[name]
        self.name = name
        self.questions = loader()
        for q in self.questions:
            if question_key != "question":
                q["question"] = q.pop(question_key)

    def gold(self, question: dict) -> str:
        return str(question[self.answer_key])

    @staticmethod
    def extract_answer(output: str) -> str:
        try:
            return remove_boxed(last_boxed_only_string(output))
        except Exception:
            return ""

    def evaluate_responses(self, responses: list[dict]) -> list[bool]:
        """responses[i] = {"prediction": str, ...} for question i, in benchmark order."""
        assert len(self.questions) == len(responses), (len(self.questions), len(responses))
        return [is_equiv(self.gold(q), r["prediction"]) for q, r in zip(self.questions, responses)]


def load_benchmark(name: str) -> Benchmark:
    return Benchmark(name)
