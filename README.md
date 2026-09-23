# Planned Test-Time Scaling (PTTS)

Code structure:

```
verl/                      git submodule: verl fork with the planner trainer (github.com/shirley-wu/verl-ptts)
rl/prompts.py              every prompt (planner, executor, plain solver); identical copy in eval/prompts.py
rl/prepare_data.py         builds the training / validation parquet files
rl/train_1.7b.sh           planner RL, paper recipe (Qwen3-1.7B-Base planner, frozen Qwen3-1.7B executor)
rl/train_4b.sh             same with Qwen3-4B-Base / Qwen3-4B
rl/math_reward.py          reward function verl uses to score executor solutions
eval/prompts.py            identical to rl/prompts.py (check with `diff rl/prompts.py eval/prompts.py`)
eval/ptts_infer.py         planner -> executor inference
eval/repeated_sampling.py  repeated-sampling baseline
eval/compute_pass_k.py     pass@k over stored samples
eval/evalchemy.py          benchmark questions + grading (MATH-500, AIME-2024/25/26, AMC-23, HMMT-2026)
```

## Setup

```bash
git clone --recurse-submodules git@github.com:shirley-wu/planned-test-time-scaling.git
cd planned-test-time-scaling
pip install -e verl            # verl fork (see verl/README.md for its own requirements: vllm, ray, ...)
pip install openai tqdm datasets lm_eval
```

## Training the planner

```bash
python rl/prepare_data.py                       # -> data/dapo-math-17k.parquet, data/test.parquet
bash rl/train_1.7b.sh                           # or rl/train_4b.sh; 8 GPUs by default
```

Training data is DAPO-Math-17k (`open-r1/DAPO-Math-17k-Processed`); validation is AIME-2024 + AMC-23. Every hyperparameter is in the training script; `N_GPUS`, `DATA_DIR`, `CKPT_DIR`, `LOGGER`, `TOTAL_STEPS` can be overridden through the environment. The paper checkpoints are `global_step_240`. Export a checkpoint to HuggingFace format before evaluating it:

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir checkpoints/ptts-planner-1.7b/global_step_240/actor \
    --target_dir checkpoints/ptts-planner-1.7b/global_step_240/actor/huggingface
```

## Evaluation

```bash
# PTTS: 64 solutions per question = 16 planner calls x 4 outlines, executor up to 10000 tokens, T = 0.7
python eval/ptts_infer.py \
    --planner-model-path checkpoints/ptts-planner-1.7b/global_step_240/actor/huggingface \
    --executor-model-path Qwen/Qwen3-1.7B \
    --datasets AIME-2024 AIME-2025 AIME-2026 MATH-500 HMMT-2026 \
    --save-dir outputs/ptts --gpu-indices 0 1 2 3
python eval/compute_pass_k.py outputs/ptts/<planner_name>/responses/separate --datasets AIME-2024 AIME-2025 AIME-2026

# Repeated-sampling baseline: 64 samples per question, up to 10000 tokens, T = 0.7
python eval/repeated_sampling.py --model-path Qwen/Qwen3-1.7B \
    --datasets AIME-2024 AIME-2025 AIME-2026 MATH-500 HMMT-2026 \
    --save-dir outputs/rs --gpu-indices 0 1 2 3
python eval/compute_pass_k.py outputs/rs/Qwen__Qwen3-1.7B --datasets AIME-2024 AIME-2025 AIME-2026
```

Both scripts start their own `vllm serve` (one data-parallel replica per GPU in `--gpu-indices`), are resumable, and write one `<dataset>/<i>.jsonl` file per sample `i` with a `{"response", "prediction"}` line per question. `compute_pass_k.py` re-grades the stored predictions and reports the unbiased pass@k estimator averaged over questions. An untrained planner is evaluated by passing a base model (e.g. `Qwen/Qwen3-1.7B-Base`) as `--planner-model-path`; other executors (e.g. `CMU-AIRe/e3-1.7B`) by changing `--executor-model-path` / `--model-path`.
