"""Build the parquet files consumed by rl/train_1.7b.sh and rl/train_4b.sh.

  data/dapo-math-17k.parquet  train: open-r1/DAPO-Math-17k-Processed (en, train split)
  data/test.parquet           val:   AIME-2024 + AMC-23 (from eval/evalchemy_data)

Each row has the keys read by the verl outline trainer:
  data_source, reward_model{ground_truth, style}, extra_info{index}
  prompt           planner chat prompt (prompts.planner_messages)
  solution_prompt  executor chat prompt whose user turn contains the literal "{outline}" placeholder;
                   the trainer substitutes each generated outline into it at rollout time.

Usage:
  python rl/prepare_data.py [--output-dir data] [--num-outlines 4] [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

from prompts import OUTLINE_PLACEHOLDER, executor_messages, planner_messages

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DATA_DIR = REPO_ROOT / "eval" / "evalchemy_data"

TRAIN_DATASET = "open-r1/DAPO-Math-17k-Processed"
# data_source -> (file under eval/evalchemy_data, question key, answer key)
VAL_SETS = {
    "aime-2024": ("aime24.json", "problem", "expected_answer"),
    "amc-23": ("amc23.json", "question", "answer"),
}


def make_row(question: str, answer: str, data_source: str, index: str, num_outlines: int) -> dict:
    return {
        "data_source": data_source,
        "reward_model": {"ground_truth": str(answer), "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": str(index)},
        "prompt": planner_messages(question, num_outlines),
        "solution_prompt": executor_messages(question, OUTLINE_PLACEHOLDER),
    }


def build_train_rows(num_outlines: int, limit: int | None) -> list[dict]:
    ds = load_dataset(TRAIN_DATASET, data_dir="en", split="train", verification_mode="no_checks")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        make_row(
            question=ex["prompt"],
            answer=ex["solution"],
            data_source="math_dapo",
            index=(ex.get("extra_info") or {}).get("index", i),
            num_outlines=num_outlines,
        )
        for i, ex in enumerate(ds)
    ]


def build_val_rows(num_outlines: int, limit: int | None) -> list[dict]:
    rows = []
    for data_source, (filename, question_key, answer_key) in VAL_SETS.items():
        with open(EVAL_DATA_DIR / filename) as f:
            examples = [json.loads(line) for line in f if line.strip()]
        if limit is not None:
            examples = examples[:limit]
        for i, ex in enumerate(examples):
            rows.append(
                make_row(
                    question=ex[question_key],
                    answer=ex[answer_key],
                    data_source=data_source,
                    index=f"{data_source}-{i}",
                    num_outlines=num_outlines,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--num-outlines", default=4, type=int)
    parser.add_argument("--limit", default=None, type=int, help="keep only the first N rows (smoke test)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_train_rows(args.num_outlines, args.limit)
    train_path = out_dir / "dapo-math-17k.parquet"
    Dataset.from_list(train_rows).to_parquet(str(train_path))
    print(f"wrote {len(train_rows)} train rows to {train_path}")

    val_rows = build_val_rows(args.num_outlines, args.limit)
    val_path = out_dir / "test.parquet"
    Dataset.from_list(val_rows).to_parquet(str(val_path))
    print(f"wrote {len(val_rows)} val rows to {val_path}")

    print("example planner prompt:\n" + train_rows[0]["prompt"][0]["content"])
    print("example executor prompt (user turn):\n" + train_rows[0]["solution_prompt"][1]["content"])


if __name__ == "__main__":
    main()
