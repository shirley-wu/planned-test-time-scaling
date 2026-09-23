"""Planned test-time scaling (PTTS) inference: planner writes outlines, executor solves each one.

For every question, the planner is called k / num_outlines times ("outline groups"); each call yields
num_outlines numbered outlines, and the executor solves the problem once per outline, so every question
ends up with k solutions. The two models are served one after the other with vLLM.

Outputs under <save-dir>/<planner_name>/:
  outlines/<dataset>/<g>.jsonl              one line per question: response_outlines, outlines[], ...
  responses/<dataset>/<g>.jsonl             same plus response[] / prediction[] (one per outline)
  responses/separate/<dataset>/<i>.jsonl    flattened, i = g * num_outlines + j, {"response", "prediction"}
                                            -> pass eval/compute_pass_k.py the responses/separate directory
  outlines/config.json, responses/config.json, */vllm.log

A planner output that cannot be parsed into exactly num_outlines outlines (after --max-outline-retries
retries) is recorded with outline_parse_failed=true and counts as wrong for all its branches.
Complete outputs are skipped on rerun.

Paper protocol (defaults): --k 64 --num-outlines 4 --outline-max-tokens 4096 --solution-max-tokens 10000
                           --outline-temperature 0.7 --solution-temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evalchemy import DATASET_CHOICES, load_benchmark
from prompts import executor_messages, planner_messages
from vllm_client import VLLMClient

SERVER_RESTART_SLEEP_SECONDS = 60

DUMMY_FAILED_OUTLINE = "Outline parsing failed; no solution branch was generated."
DUMMY_FAILED_RESPONSE = "Outline parsing failed; no answer generated."
DUMMY_FAILED_PREDICTION = ""


def sanitize_model_name(model_path: str) -> str:
    return model_path.strip("/").replace("/", "__")


def parse_outlines(text: str, num_outlines: int) -> list[str]:
    """Split "1. ... 2. ... N. ..." into N outlines; raise if any marker is missing or duplicated."""
    text = (text or "").strip()
    if "1. " in text and not text.startswith("1. "):
        text = text[text.index("1. "):].strip()
    elif not text.startswith("1. "):
        text = f"1. {text}"

    numbered_text = f"\n{text}"
    outlines = []
    for index in range(1, num_outlines + 1):
        marker = f"\n{index}. "
        if numbered_text.count(marker) != 1:
            raise ValueError(f"expected exactly one outline marker: {index}.")
        start = numbered_text.index(marker) + len(marker)
        end = numbered_text.find(f"\n{index + 1}. ", start)
        outline = numbered_text[start:end if end != -1 else None].strip()
        if not outline:
            raise ValueError(f"outline {index} is empty")
        outlines.append(outline)
    return outlines


# ----------------------------------------------------------------------------------------------- model calls

def generate_outline(task: dict, client: VLLMClient, args) -> tuple[int, int, dict]:
    messages = planner_messages(task["question"]["question"], args.num_outlines)
    raw_response, last_error = "", None
    for retry in range(args.max_outline_retries + 1):
        try:
            response = client.openai_client().chat.completions.create(
                model=args.planner_model_name,
                messages=messages,
                temperature=args.outline_temperature,
                max_tokens=args.outline_max_tokens,
            )
            raw_response = response.choices[0].message.content or ""
            outlines = parse_outlines(raw_response, args.num_outlines)
            row = {"response_outlines": raw_response, "outlines": outlines, "outline_parse_failed": False}
            break
        except Exception as exc:  # parse failure or request error -> retry
            last_error = exc
    else:
        row = {
            "response_outlines": raw_response or f"Outline generation failed: {last_error}",
            "outlines": [DUMMY_FAILED_OUTLINE] * args.num_outlines,
            "outline_parse_failed": True,
        }
        retry = args.max_outline_retries
    row["num_outlines"] = args.num_outlines
    row["outline_parse_retry"] = retry
    return task["group_idx"], task["question_idx"], row


def generate_solution(task: dict, client: VLLMClient, benchmark, args) -> tuple[int, int, int, str, str]:
    response = client.openai_client().chat.completions.create(
        model=args.executor_model_name,
        messages=executor_messages(task["question"]["question"], task["outline"]),
        temperature=args.solution_temperature,
        max_tokens=args.solution_max_tokens,
    )
    text = response.choices[0].message.content or ""
    return task["group_idx"], task["question_idx"], task["outline_idx"], text, benchmark.extract_answer(text)


# ----------------------------------------------------------------------------------------------- file helpers

def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _rows_or_none(path: Path) -> list[dict] | None:
    try:
        return load_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return None


def outline_file_complete(path: Path, num_questions: int, num_outlines: int) -> bool:
    rows = _rows_or_none(path)
    return rows is not None and len(rows) == num_questions and all(
        row.get("num_outlines") == num_outlines
        and len(row.get("outlines", [])) == num_outlines
        and "response_outlines" in row
        and "outline_parse_failed" in row
        for row in rows
    )


def response_file_complete(path: Path, num_questions: int, num_outlines: int) -> bool:
    rows = _rows_or_none(path)
    return rows is not None and len(rows) == num_questions and all(
        len(row.get("outlines", [])) == num_outlines
        and len(row.get("response", [])) == num_outlines
        and len(row.get("prediction", [])) == num_outlines
        for row in rows
    )


def separate_file_complete(path: Path, num_questions: int) -> bool:
    rows = _rows_or_none(path)
    return rows is not None and len(rows) == num_questions and all("response" in r and "prediction" in r for r in rows)


def outline_dataset_complete(outline_root: Path, dataset_name: str, args, num_questions: int) -> bool:
    return all(
        outline_file_complete(outline_root / dataset_name / f"{g}.jsonl", num_questions, args.num_outlines)
        for g in range(args.num_outline_groups)
    )


def response_dataset_complete(response_root: Path, dataset_name: str, args, num_questions: int) -> bool:
    return all(
        response_file_complete(response_root / dataset_name / f"{g}.jsonl", num_questions, args.num_outlines)
        for g in range(args.num_outline_groups)
    ) and all(
        separate_file_complete(response_root / "separate" / dataset_name / f"{g * args.num_outlines + j}.jsonl", num_questions)
        for g in range(args.num_outline_groups)
        for j in range(args.num_outlines)
    )


# ----------------------------------------------------------------------------------------------- stages

def start_server(model_path: str, model_name: str, log_path: Path, args) -> VLLMClient:
    client = VLLMClient(model_path, model_name, port=args.port, host=args.host, api_key=args.api_key, request_timeout_s=args.timeout)
    client.start_server(
        data_parallel_size=len(args.gpu_indices),
        gpu_memory_utilization=args.gpu_memory_utilization,
        cuda_visible_devices=",".join(args.gpu_indices),
        log_path=log_path,
        startup_timeout_s=args.startup_timeout,
    )
    return client


def run_planner_dataset(dataset_name: str, benchmark, client: VLLMClient, args, outline_root: Path) -> None:
    questions = benchmark.questions
    if outline_dataset_complete(outline_root, dataset_name, args, len(questions)):
        print(f"{dataset_name}: outlines complete, skipping")
        return

    outputs = [[None] * len(questions) for _ in range(args.num_outline_groups)]
    tasks = [
        {"group_idx": g, "question_idx": q, "question": question}
        for g in range(args.num_outline_groups)
        for q, question in enumerate(questions)
    ]
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(generate_outline, task, client, args) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset_name} outlines"):
            g, q, row = future.result()
            outputs[g][q] = row

    for g, rows in enumerate(outputs):
        assert all(r is not None for r in rows)
        write_jsonl(outline_root / dataset_name / f"{g}.jsonl", rows)
    print(f"{dataset_name}: wrote outlines to {outline_root / dataset_name}")


def run_executor_dataset(dataset_name: str, benchmark, client: VLLMClient, args, outline_root: Path, response_root: Path) -> None:
    questions = benchmark.questions
    if response_dataset_complete(response_root, dataset_name, args, len(questions)):
        print(f"{dataset_name}: responses complete, skipping")
        return

    # one branch per (group, question, outline)
    branches: dict[tuple[int, int, int], dict] = {}
    for g in range(args.num_outline_groups):
        path = outline_root / dataset_name / f"{g}.jsonl"
        if not outline_file_complete(path, len(questions), args.num_outlines):
            raise RuntimeError(f"outline file is missing or incomplete: {path}")
        for q, outline_row in enumerate(load_jsonl(path)):
            for j, outline in enumerate(outline_row["outlines"]):
                failed = outline_row["outline_parse_failed"]
                branches[(g, q, j)] = {
                    **outline_row,
                    "outline": outline,
                    "question": questions[q],
                    "response": DUMMY_FAILED_RESPONSE if failed else "",
                    "prediction": DUMMY_FAILED_PREDICTION if failed else "",
                }

    tasks = [
        {"group_idx": g, "question_idx": q, "outline_idx": j, **branch}
        for (g, q, j), branch in branches.items()
        if not branch["outline_parse_failed"]
    ]
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(generate_solution, task, client, benchmark, args) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset_name} execute"):
            g, q, j, text, prediction = future.result()
            branches[(g, q, j)]["response"] = text
            branches[(g, q, j)]["prediction"] = prediction

    for g in range(args.num_outline_groups):
        group_rows = []
        for q in range(len(questions)):
            subs = [branches[(g, q, j)] for j in range(args.num_outlines)]
            group_rows.append(
                {
                    "response_outlines": subs[0]["response_outlines"],
                    "outlines": [s["outline"] for s in subs],
                    "num_outlines": args.num_outlines,
                    "outline_parse_failed": subs[0]["outline_parse_failed"],
                    "outline_parse_retry": subs[0]["outline_parse_retry"],
                    "response": [s["response"] for s in subs],
                    "prediction": [s["prediction"] for s in subs],
                }
            )
        write_jsonl(response_root / dataset_name / f"{g}.jsonl", group_rows)
        for j in range(args.num_outlines):
            write_jsonl(
                response_root / "separate" / dataset_name / f"{g * args.num_outlines + j}.jsonl",
                [{"response": row["response"][j], "prediction": row["prediction"][j]} for row in group_rows],
            )
    print(f"{dataset_name}: wrote responses to {response_root}")


def write_config(args, outline_root: Path, response_root: Path) -> None:
    config = {
        "datasets": args.datasets,
        "planner_model_path": args.planner_model_path,
        "executor_model_path": args.executor_model_path,
        "prompt_version": "v0",
        "k": args.k,
        "num_outlines": args.num_outlines,
        "num_outline_groups": args.num_outline_groups,
        "outline_max_tokens": args.outline_max_tokens,
        "solution_max_tokens": args.solution_max_tokens,
        "outline_temperature": args.outline_temperature,
        "solution_temperature": args.solution_temperature,
        "max_outline_retries": args.max_outline_retries,
        "num_workers": args.num_workers,
    }
    for root in (outline_root, response_root):
        root.mkdir(parents=True, exist_ok=True)
        with (root / "config.json").open("w") as f:
            json.dump(config, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-model-path", required=True)
    parser.add_argument("--executor-model-path", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--gpu-indices", nargs="+", required=True, help="one vLLM data-parallel replica per GPU")
    parser.add_argument("--k", type=int, default=64, help="solutions per question (rounded up to a multiple of --num-outlines)")
    parser.add_argument("--num-outlines", type=int, default=4, help="outlines per planner call")
    parser.add_argument("--outline-max-tokens", type=int, default=4096)
    parser.add_argument("--solution-max-tokens", type=int, default=10000)
    parser.add_argument("--outline-temperature", type=float, default=0.7)
    parser.add_argument("--solution-temperature", type=float, default=0.7)
    parser.add_argument("--max-outline-retries", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=int, default=3600, help="per-request timeout (s)")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()

    if args.k < 1 or args.num_outlines < 1:
        parser.error("--k and --num-outlines must be >= 1")
    if args.max_outline_retries < 0:
        parser.error("--max-outline-retries must be >= 0")
    args.num_outline_groups = -(-args.k // args.num_outlines)  # ceil
    args.k = args.num_outline_groups * args.num_outlines
    args.planner_model_name = sanitize_model_name(args.planner_model_path)
    args.executor_model_name = sanitize_model_name(args.executor_model_path)
    return args


def main() -> None:
    args = parse_args()
    outline_root = Path(args.save_dir) / args.planner_model_name / "outlines"
    response_root = Path(args.save_dir) / args.planner_model_name / "responses"
    write_config(args, outline_root, response_root)

    benchmarks = {name: load_benchmark(name) for name in args.datasets}
    need_responses = [n for n, b in benchmarks.items() if not response_dataset_complete(response_root, n, args, len(b.questions))]
    need_outlines = [n for n in need_responses if not outline_dataset_complete(outline_root, n, args, len(benchmarks[n].questions))]

    if need_outlines:
        client = None
        try:
            client = start_server(args.planner_model_path, args.planner_model_name, outline_root / "vllm.log", args)
            for name in need_outlines:
                run_planner_dataset(name, benchmarks[name], client, args, outline_root)
        finally:
            if client is not None:
                client.stop_server()
        print(f"sleeping {SERVER_RESTART_SLEEP_SECONDS}s before starting the executor server")
        time.sleep(SERVER_RESTART_SLEEP_SECONDS)

    if not need_responses:
        print("all requested responses are complete")
        return

    client = None
    try:
        client = start_server(args.executor_model_path, args.executor_model_name, response_root / "vllm.log", args)
        for name in need_responses:
            run_executor_dataset(name, benchmarks[name], client, args, outline_root, response_root)
    finally:
        if client is not None:
            client.stop_server()


if __name__ == "__main__":
    main()
