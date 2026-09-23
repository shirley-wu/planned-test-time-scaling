"""All prompts used in this repo.

Two identical copies exist: rl/prompts.py and eval/prompts.py (keep them in sync: `diff rl/prompts.py eval/prompts.py`).
Used by rl/prepare_data.py (training data), eval/ptts_infer.py (planner + executor inference)
and eval/repeated_sampling.py (baseline). Keep training and evaluation prompts identical.
"""

# Planner: writes {num_outlines} high-level solution outlines as a numbered list "1. ... N. ...".
PLANNER_SYSTEM_PROMPT = """You are an annotator tasked with generating multiple high-level solution outlines for a math problem.

Your goal is to explore different perspectives, strategies, or conceptual approaches that could be used to solve the problem. Based on this, produce {num_outlines} distinct outlines that could independently guide a solver from start to finish.
* You must NOT solve the problem.
* You must NOT compute values, simplify expressions, or use algebra.
* Any calculation makes the output invalid.

Format output as a numbered list (1. - {num_outlines}.), where each item is an outline."""

# Executor: solves the problem following one outline.
EXECUTOR_SYSTEM_PROMPT = (
    "You are a math problem solver. You are given a math problem and a solution outline. "
    "Follow the outline carefully to solve the problem step by step. "
    "Show your work and put your final answer in \\boxed{}."
)

EXECUTOR_USER_PROMPT = """Problem:
{question}

Solution Outline:
{outline}

Now solve the problem by following this outline:"""

# Placeholder left in the training data's `solution_prompt`; the verl outline trainer substitutes each
# generated outline into it at rollout time.
OUTLINE_PLACEHOLDER = "{outline}"

# Repeated-sampling baseline: plain solver, no outline.
SOLVER_SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step. "
    "Show your work and put your final answer in \\boxed{}."
)


def planner_messages(question: str, num_outlines: int = 4) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT.format(num_outlines=num_outlines)},
        {"role": "user", "content": question},
    ]


def executor_messages(question: str, outline: str = OUTLINE_PLACEHOLDER) -> list[dict[str, str]]:
    # str.replace rather than str.format: questions and outlines contain LaTeX braces.
    user = EXECUTOR_USER_PROMPT.replace("{question}", question).replace("{outline}", outline)
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def solver_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
