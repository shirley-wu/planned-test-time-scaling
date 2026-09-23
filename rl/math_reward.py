"""Reward function handed to verl via `custom_reward_function.path/name` (see rl/train_1.7b.sh).

It scores ONE executor solution: 1.0 if the last \\boxed{} answer matches the ground truth under verl's
DAPO math checker (strict boxed extraction), else 0.0. The outline trainer then rewards each outline
with the max score over its executor solutions (verl/trainer/ppo/ray_trainer_outline.py).
"""


def reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    from verl.utils.reward_score import math_dapo

    res = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    return float(res["acc"])
