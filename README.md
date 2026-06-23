# WebNav-RL

WebNav-RL is a local, verifiable mini web-navigation project for studying small-model agentic post-training. It builds a deterministic browser-like environment, generates synthetic navigation tasks, creates expert trajectories, trains a LoRA SFT adapter on Qwen2.5-0.5B-Instruct, evaluates full tool-use rollouts, performs error analysis, designs a reward function, and prototypes a GRPO-style update loop.

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
-> minimal GRPO prototype
```

## Current Status

The latest stable policy is the 200-step LoRA SFT adapter:

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Local model path: `models/qwen2.5-0.5b-instruct`
- SFT adapter: `outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200`
- Full 200-task eval success rate: `63.5%`
- Tool-call format accuracy after SFT: `100%`
- Invalid tool-call rate in full SFT eval: `1.88%`

The GRPO-style trainer is currently a proof of concept. It proves that group rollouts, reward scoring, advantage computation, and LoRA updates are wired together, but the tiny prototype run regressed on eval and should not be presented as a final improvement.

## Key Results

| Stage | Eval | Result | Interpretation |
| --- | --- | --- | --- |
| Base Qwen first-step eval | 20 tasks, max 1 step | `0.0` format accuracy | Base chat model does not naturally emit the required XML/JSON tool-call format. |
| SFT step200 first-step eval | 200 tasks, max 1 step | `1.0` format accuracy | LoRA SFT solves the action-format problem. |
| SFT step200 full rollout | 200 tasks | `127/200 = 63.5%` success | The model can complete many multi-step navigation tasks locally. |
| SFT error analysis | 200 tasks | 51 wrong click paths, 18 wrong candidates, 4 invalid calls | Remaining failures are mostly decision quality, not format. |
| Reward scoring | SFT 200-task trajectories | success mean `0.876`, failure mean `0.221` | Reward separates good and bad rollouts enough for RL experiments. |
| GRPO group rollout | 4 tasks x 4 samples | mean reward `0.566`, nonzero advantages in 3/4 groups | Sampling creates useful within-group preference signal. |
| Minimal GRPO prototype | 20 tasks | `45%` success | Training loop works, but tiny/no-KL prototype is not yet an improvement. |

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
| GRPO prototype | `training/grpo_train.py` | Runs a minimal advantage-weighted LoRA update. |

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

`models/` and `.python_deps/` are intentionally ignored by git.

## Quickstart

Run the deterministic V0/V1 data pipeline:

```bash
python scripts/run_v1_data.py --num-tasks 1000 --train-ratio 0.8 --seed 7
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
- `docs/SERVER_RUNBOOK.md`: server upload and GRPO-KL experiment runbook.
- `docs/PROJECT_SUMMARY.md`: Chinese project summary for interview preparation.

## Interview Summary

The strongest way to describe the project is:

> I built a small but complete local web-navigation post-training pipeline. It starts from deterministic page/task generation, creates expert tool-use trajectories, trains a small Qwen model with LoRA SFT, evaluates real multi-step tool interaction, analyzes failure modes, designs a shaped reward, and then prototypes the GRPO data/training loop. The SFT stage improved the model from failing the tool-call format entirely to 100% format accuracy and 63.5% full-task success on 200 held-out tasks. The current RL stage is intentionally labeled as a prototype because the minimal no-KL update regressed, but the infrastructure for group rollouts and advantage computation is now in place.

## Next Work

- Add more diverse tasks and pages to reduce overfitting to element IDs and page templates.
- Improve candidate selection for sorted/filtered result lists.
- Add KL/reference-policy control to the GRPO trainer.
- Scale group rollouts beyond the 4-task smoke setting.
- Compare SFT-only, reward-reranking, and GRPO-updated adapters on the same full eval split.
