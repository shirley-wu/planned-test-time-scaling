"""pass@k from stored samples.

Reads <DIR>/<dataset>/<i>.jsonl for i in 0..n-1 (one line per question, in benchmark order, each with a
"prediction" field), re-grades every prediction, and reports the unbiased estimator

    pass@k = 1 - C(n - c, k) / C(n, k)      (n samples, c correct)   averaged over questions.

<DIR> is  <save-dir>/<model_name>                       for eval/repeated_sampling.py
      or  <save-dir>/<planner_name>/responses/separate  for eval/ptts_infer.py

Usage:
  python eval/compute_pass_k.py DIR [--datasets AIME-2024 AIME-2025 AIME-2026] [--n 64] [--ks 1 4 8 16 32 64]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from evalchemy import DATASET_CHOICES, load_benchmark


def load_accs(d: str, dataset: str, n: int) -> np.ndarray:
    """Return a bool array of shape (n, num_questions)."""
    benchmark = load_benchmark(dataset)
    accs = []
    for i in range(n):
        path = os.path.join(d, dataset, f"{i}.jsonl")
        assert os.path.exists(path), path
        with open(path) as f:
            outputs = [json.loads(line) for line in f]
        accs.append(benchmark.evaluate_responses(outputs))
    return np.array(accs, dtype=bool)


def pass_at_k(k: int, accs: np.ndarray) -> float:
    """Unbiased pass@k for one question given its sample correctness vector."""
    n = len(accs)
    c = int(accs.sum())
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def pass_k_table(d: str, datasets: list[str], n: int, ks: list[int]) -> dict[str, dict[int, float]]:
    """{dataset: {k: pass@k in percent}} plus an "Average" row over datasets."""
    table: dict[str, dict[int, float]] = {}
    for dataset in datasets:
        accs = load_accs(d, dataset, n)
        table[dataset] = {k: 100 * float(np.mean([pass_at_k(k, accs[:, q]) for q in range(accs.shape[1])])) for k in ks}
    if len(datasets) > 1:
        table["Average"] = {k: float(np.mean([table[ds][k] for ds in datasets])) for k in ks}
    return table


def format_markdown(table: dict[str, dict[int, float]], ks: list[int]) -> str:
    lines = ["| dataset | " + " | ".join(f"pass@{k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for name, row in table.items():
        lines.append(f"| {name} | " + " | ".join(f"{row[k]:.1f}" for k in ks) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=["AIME-2024", "AIME-2025", "AIME-2026"])
    parser.add_argument("--n", type=int, default=64, help="number of sample files per dataset")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64])
    args = parser.parse_args()
    assert max(args.ks) <= args.n, "every k must be <= n"

    print(format_markdown(pass_k_table(args.dir, args.datasets, args.n, args.ks), args.ks))


if __name__ == "__main__":
    main()
