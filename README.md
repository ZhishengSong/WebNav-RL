# WebNav-RL

WebNav-RL is a local, verifiable mini web-navigation project for studying small-model agentic post-training. It builds a deterministic browser-like environment, generates synthetic navigation tasks, creates expert trajectories, trains a LoRA SFT adapter on Qwen2.5-0.5B-Instruct, evaluates full tool-use rollouts, performs error analysis, designs a reward function, and implements a GRPO-style update loop with reference-policy KL.

The project is intentionally small enough to run on a local laptop, while still covering the main research pipeline:

```text
local web pages
-> tasks
-> expert trajectories
-> SFT data
-> base model eval
-> LoRA SFT
-> full rollout eval
-> error analysis
-> reward design
-> GRPO group rollouts
-> reference-policy KL training
-> multi-seed paired evaluation
```

## Current Status

The controlled server experiment uses the 200-step LoRA SFT adapter as its stable baseline:

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Local model path: `models/qwen2.5-0.5b-instruct`
- SFT adapter: `outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200`
- Server SFT baseline: `121/200 = 60.5%`
- Best observed GRPO-KL run: `128/200 = 64.0%`
- Three-seed GRPO-KL mean: `62.17% +/- 2.36 pp`
- Tool-call format accuracy: `100%` for SFT and all GRPO-KL runs

The best GRPO-KL run improved success by `+3.5 pp`, but the three training seeds scored `64.0%`, `63.0%`, and `59.5%`. The mean improvement is positive (`+1.67 pp`) but seed variance remains substantial, so the result is reported as preliminary positive evidence rather than a stable significant gain.

V2 environment/data generation is now ready for the next training round. It adds randomized visible element IDs, two seen train layouts, one structurally held-out eval layout, balanced targeted comparison tasks, and 3,500 expert-verified trajectories. No V2 model result is claimed yet.

## Key Results

| Stage | Eval | Result | Interpretation |
| --- | --- | --- | --- |
| Base Qwen first-step eval | 20 tasks, max 1 step | `0.0` format accuracy | Base chat model does not naturally emit the required XML/JSON tool-call format. |
| SFT step200 first-step eval | 200 tasks, max 1 step | `1.0` format accuracy | LoRA SFT solves the action-format problem. |
| Local SFT step200 run | 200 tasks | `127/200 = 63.5%` success | A separate local run established multi-step navigation capability; it is not the server GRPO baseline. |
| SFT error analysis | 200 tasks | 51 wrong click paths, 18 wrong candidates, 4 invalid calls | Remaining failures are mostly decision quality, not format. |
| Reward scoring | SFT 200-task trajectories | success mean `0.876`, failure mean `0.221` | Reward separates good and bad rollouts enough for RL experiments. |
| GRPO group rollout | 4 tasks x 4 samples | mean reward `0.566`, nonzero advantages in 3/4 groups | Sampling creates useful within-group preference signal. |
| Minimal GRPO prototype | 20 tasks | `45%` success | Training loop works, but tiny/no-KL prototype is not yet an improvement. |
| Server GRPO-KL | 3 training seeds, 200-task eval | best `64.0%`, mean `62.17%` vs SFT `60.5%` | Positive mean signal, but seed variance remains substantial. |
| V2 data readiness | 3,000 train + 500 held-out eval | `3,500/3,500` expert success, zero train/eval ID overlap | Tests structural generalization instead of fixed-ID memorization. |

## Main Components

| Area | Files | Purpose |
| --- | --- | --- |
| Page generation | `pages/page_generator.py` | Creates local shopping/course HTML pages plus metadata. |
| Environment | `env/browser_env.py`, `env/verifier.py` | Executes tool actions and verifies final answers. |
| Tools | `tools/tool_registry.py`, `tools/web_tools.py` | Defines the callable action surface. |
| Task/data generation | `tasks/task_generator.py`, `scripts/run_v1_data.py` | Produces train/eval tasks and expert trajectories. |
| SFT data | `training/build_sft_data.py` | Converts expert trajectories to chat-style SFT examples. |
| Model eval | `rollout/model_runner.py`, `rollout/transformers_generator.py`, `scripts/run_eval.py` | Runs real model/tool interaction loops. |
| SFT training | `training/sft_train.py` | Trains a LoRA adapter on expert next-action data. |
| Analysis | `eval/error_analysis.py` | Labels failures by behavior type. |
| Reward | `training/reward.py` | Scores trajectories for RL-style training. |
| GRPO rollout | `training/grpo_rollout.py` | Samples grouped rollouts and computes group-relative advantages. |
| GRPO-KL training | `training/grpo_train.py` | Runs advantage-weighted LoRA updates with a frozen reference-policy KL penalty. |
| Multi-seed analysis | `eval/multiseed_analysis.py` | Computes paired transitions, McNemar tests, seed aggregates, and template-level deltas. |
| V2 pages | `pages/v2_generator.py` | Builds three layouts with random visible element IDs and distractors. |
| V2 tasks | `tasks/v2_task_generator.py` | Creates balanced targeted tasks with a held-out structural split. |
| V2 pipeline | `scripts/run_v2_data.py` | Generates pages/tasks, verifies expert trajectories, and builds SFT data. |

## Important Artifacts

| Artifact | Path |
| --- | --- |
| Train SFT data | `training/sft_train.jsonl` |
| Eval SFT data | `training/sft_eval.jsonl` |
| Full SFT eval report | `outputs/eval_reports/sft_qwen_0_5b_step200_eval200_full_t48_report.json` |
| Full SFT eval trajectories | `outputs/trajectories/sft_qwen_0_5b_step200_eval200_full_t48_trajectories.jsonl` |
| Error analysis report | `outputs/eval_reports/sft_qwen_0_5b_step200_eval200_error_analysis.json` |
| Reward report | `outputs/eval_reports/sft_qwen_0_5b_step200_eval200_reward_report.json` |
| GRPO group rollout | `outputs/rollouts/grpo_sft_step200_group4_task4.jsonl` |
| GRPO prototype adapter | `outputs/checkpoints/qwen2_5_0_5b_lora_grpo_proto_step5` |
| Final multi-seed analysis | `outputs/eval_reports/grpo_multiseed_analysis.md` |
| Downloaded server artifacts | `artifacts/server_runs/2026-06-24` |

`models/` and `.python_deps/` are intentionally ignored by git.

## Quickstart

Run the deterministic V0/V1 data pipeline:

```bash
python scripts/run_v1_data.py --num-tasks 1000 --train-ratio 0.8 --seed 7
```

Generate and verify the V2 structural-generalization dataset:

```bash
python scripts/run_v2_data.py --train-tasks 3000 --eval-tasks 500 --seed 31
```

Run unit tests:

```bash
python -m pytest -q
```

Run the zero-dependency expert eval smoke test:

```bash
python scripts/run_eval.py
```

## Local Model Environment

On this machine, model runs use:

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py --model models\qwen2.5-0.5b-instruct --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step200 --tasks tasks\eval_tasks.jsonl --limit 200 --output outputs\trajectories\sft_qwen_0_5b_step200_eval200_full_t48_trajectories.jsonl --report outputs\eval_reports\sft_qwen_0_5b_step200_eval200_full_t48_report.json --failures outputs\eval_reports\sft_qwen_0_5b_step200_eval200_full_t48_failures.jsonl --device auto --max-new-tokens 48 --temperature 0.0 --incremental --resume --report-every 10
```

The `PYTHONPATH` order matters because `.python_deps/research` contains a CPU Torch wheel, while the Conda research environment contains CUDA Torch.

## Documentation

The detailed step-by-step notes are in `docs/`:

- `docs/IMPLEMENTATION_NOTES.md`: full V0/V1 implementation explanation and interview talking points.
- `docs/STEP_01_BASE_EVAL_READINESS.md`: model evaluation framework.
- `docs/STEP_02_HF_QWEN_BASELINE.md`: Hugging Face Qwen setup and base-model baseline.
- `docs/STEP_03_LORA_SFT.md`: LoRA SFT script and first adapter.
- `docs/STEP_04_FULLER_SFT_EVAL.md`: larger SFT run and eval.
- `docs/STEP_05_INCREMENTAL_EVAL_ERROR_ANALYSIS.md`: resumable eval and failure taxonomy.
- `docs/STEP_06_REWARD_FUNCTION.md`: reward design and scoring report.
- `docs/STEP_07_GRPO_GROUP_ROLLOUT.md`: grouped rollouts and advantages.
- `docs/STEP_08_MINIMAL_GRPO_PROTOTYPE.md`: minimal GRPO-style training prototype.
- `docs/STEP_09_GRPO_WITH_KL.md`: reference-policy KL constraint for the GRPO trainer.
- `docs/STEP_10_SERVER_GRPO_KL_EXPERIMENT.md`: server-scale GRPO-KL result and paired analysis.
- `docs/STEP_11_V2_ENVIRONMENT.md`: randomized IDs, multi-layout pages, balanced tasks, and held-out split.
- `docs/FINAL_PROJECT_REPORT.md`: consolidated final technical report.
- `docs/INTERVIEW_QA.md`: interview narrative, questions, and honest resume wording.
- `docs/SERVER_RUNBOOK.md`: server upload and GRPO-KL experiment runbook.
- `docs/PROJECT_SUMMARY.md`: Chinese project summary for interview preparation.

## Interview Summary

The strongest way to describe the project is:

> I built an end-to-end web-navigation post-training pipeline around Qwen2.5-0.5B: deterministic environment and task generation, expert trajectories, LoRA SFT, resumable tool-use evaluation, failure analysis, shaped rewards, grouped rollouts, and reference-policy KL training. In a controlled 200-task server evaluation, the SFT baseline scored 60.5%; the best GRPO-KL run scored 64.0%, while three GRPO seeds averaged 62.17%. Tool-call format accuracy stayed at 100%. I report the gain as preliminary because one seed regressed and paired tests were not significant.

## Next Work

- Train the first V2 SFT baseline and measure performance on the held-out layout C.
- Compare V2 fixed-layout and structural-generalization accuracy by template.
- Reduce the `60%` zero-advantage group rate through harder tasks or more diverse sampling.
- Add early stopping and stronger drift monitoring; full-pass training increased invalid calls without improving success.
- Re-run the improved data/reward design on a 1.5B model only after the environment is more diverse.
