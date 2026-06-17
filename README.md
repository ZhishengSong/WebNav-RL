# WebNav-RL

WebNav-RL is a local, verifiable mini web navigation environment for small-model agentic post-training.

For a detailed explanation of the completed V0/V1 implementation and interview talking points, see `docs/IMPLEMENTATION_NOTES.md`.

The V0 goal is to run a full deterministic loop without model training:

```text
generate local pages
-> generate tasks
-> execute tools
-> verify submitted answer
-> save expert trajectories
```

## V0 Components

- `pages/page_generator.py`: generates shopping and course pages plus metadata.
- `tasks/task_loader.py`: loads JSONL task files.
- `env/browser_env.py`: implements `open_page`, `click`, `get_visible_text`, and `submit_answer`.
- `env/verifier.py`: exact-match verifier.
- `rollout/rollout_runner.py`: runs the rule-based expert and saves trajectories.
- `scripts/run_v0_demo.py`: one-command smoke test.

## Run

```bash
python scripts/run_v0_demo.py
```

Expected output:

```json
{
  "num_tasks": 20,
  "success": 20,
  "success_rate": 1.0,
  "avg_steps": 3.25,
  "invalid_actions": 0
}
```

Generated artifacts:

- `pages/generated_pages/*.html`
- `pages/generated_pages/metadata.json`
- `tasks/all_tasks.jsonl`
- `tasks/train_tasks.jsonl`
- `tasks/eval_tasks.jsonl`
- `outputs/trajectories/expert_trajectories.jsonl`

## V1 Data Preparation

Generate 1000 train/eval tasks, run the expert, and convert trajectories into SFT chat data:

```bash
python scripts/run_v1_data.py --num-tasks 1000 --train-ratio 0.8 --seed 7
```

Expected summary:

```json
{
  "num_tasks": 1000,
  "train_tasks": 800,
  "eval_tasks": 200,
  "train_sft_examples": 800,
  "eval_sft_examples": 200,
  "success_rate": 1.0,
  "invalid_actions": 0,
  "avg_steps": 3.167
}
```

Additional artifacts:

- `outputs/trajectories/expert_train_trajectories.jsonl`
- `outputs/trajectories/expert_eval_trajectories.jsonl`
- `training/sft_train.jsonl`
- `training/sft_eval.jsonl`
