"""Repeated-sampling baseline: k independent samples per question from one model.

Starts a vLLM server for the model, asks it each question k times with the plain solver prompt, and writes

  <save-dir>/<model_name>/config.json
  <save-dir>/<model_name>/<dataset>/<i>.jsonl      i in 0..k-1; one {"response", "prediction"} per question

which is the layout eval/compute_pass_k.py reads. Complete sample files are skipped on rerun.

Paper protocol: --k 64 --max-tokens 10000 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evalchemy import DATASET_CHOICES, load_benchmark
from prompts import solver_messages
from vllm_client import VLLMClient


def sanitize_model_name(model_path: str) -> str:
    return model_path.strip("/").replace("/", "__")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def sample_file_complete(path: Path, num_questions: int) -> bool:
    try:
        with path.open() as f:
            rows = [json.loads(line) for line in f]
    except (OSError, json.JSONDecodeError):
        return False
    return len(rows) == num_questions and all("response" in r and "prediction" in r for r in rows)


def generate(task: dict, client: VLLMClient, benchmark, args) -> tuple[int, int, dict]:
    response = client.openai_client().chat.completions.create(
        model=args.model_name,
        messages=solver_messages(task["question"]["question"]),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    text = response.choices[0].message.content or ""
    return task["sample_idx"], task["question_idx"], {"response": text, "prediction": benchmark.extract_answer(text)}


def run_dataset(dataset_name: str, client: VLLMClient, args, out_root: Path) -> None:
    benchmark = load_benchmark(dataset_name)
    questions = benchmark.questions
    todo = [i for i in range(args.k) if not sample_file_complete(out_root / dataset_name / f"{i}.jsonl", len(questions))]
    if not todo:
        print(f"{dataset_name}: all {args.k} sample files complete, skipping")
        return

    outputs = {i: [None] * len(questions) for i in todo}
    tasks = [{"sample_idx": i, "question_idx": q, "question": question} for i in todo for q, question in enumerate(questions)]
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(generate, task, client, benchmark, args) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=dataset_name):
            i, q, row = future.result()
            outputs[i][q] = row

    for i, rows in outputs.items():
        assert all(r is not None for r in rows)
        write_jsonl(out_root / dataset_name / f"{i}.jsonl", rows)
    print(f"{dataset_name}: wrote {len(todo)} sample files to {out_root / dataset_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--gpu-indices", nargs="+", required=True, help="one vLLM data-parallel replica per GPU")
    parser.add_argument("--k", type=int, default=64, help="samples per question")
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=int, default=3600, help="per-request timeout (s)")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()
    args.model_name = sanitize_model_name(args.model_path)
    return args


def main() -> None:
    args = parse_args()
    out_root = Path(args.save_dir) / args.model_name
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "config.json").open("w") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "api_key"}, f, indent=2)

    todo = [
        ds for ds in args.datasets
        if any(not sample_file_complete(out_root / ds / f"{i}.jsonl", len(load_benchmark(ds).questions)) for i in range(args.k))
    ]
    if not todo:
        print("all requested sample files are complete")
        return

    client = VLLMClient(args.model_path, args.model_name, port=args.port, host=args.host, api_key=args.api_key, request_timeout_s=args.timeout)
    try:
        client.start_server(
            data_parallel_size=len(args.gpu_indices),
            gpu_memory_utilization=args.gpu_memory_utilization,
            cuda_visible_devices=",".join(args.gpu_indices),
            log_path=out_root / "vllm.log",
            startup_timeout_s=args.startup_timeout,
        )
        for ds in todo:
            run_dataset(ds, client, args, out_root)
    finally:
        client.stop_server()


if __name__ == "__main__":
    main()
