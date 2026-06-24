# Server Runbook: GRPO-KL Experiment

这份文档用于把项目上传到服务器后，快速跑一组更有意义的 GRPO-KL 实验。

目标流程：

```text
setup env
-> download Qwen2.5-0.5B-Instruct
-> train/recreate SFT step200 adapter
-> collect 100 train tasks x group size 4 GRPO rollouts
-> train GRPO-KL adapter
-> eval SFT and GRPO-KL on the same 200-task split
-> generate comparison report
```

## 1. 服务器建议

当前实验使用 0.5B 模型 + LoRA，不需要 A100。更重要的是节省 rollout 时间。

推荐：

- GPU: 24GB 显存更舒服，例如 RTX 4090 / L4 / A10G。
- Disk: 至少 50GB。
- Python: 3.10 或 3.11。
- CUDA: 能正常跑 PyTorch CUDA 即可。

如果只能用 16GB 显存，也可以尝试，因为模型很小；如果显存不足，先把 rollout/eval 的并发保持为当前单进程默认即可。

## 2. 上传哪些文件

建议用 git 上传代码。

不要上传：

- `.python_deps/`
- `.venv/`
- `models/`
- `outputs/checkpoints/`
- 大量 `outputs/trajectories/*.jsonl`
- 大量 `outputs/rollouts/*.jsonl`

这些内容要么可以在服务器重新生成，要么太大。

如果你想节省 SFT 训练时间，也可以单独上传本地 SFT adapter：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200
```

如果不上传，服务器脚本会用 `training/sft_train.jsonl` 重新训练一个 step200 adapter。

## 3. 基础环境

进入项目根目录后：

```bash
bash scripts/server/00_setup_env.sh
source .venv/bin/activate
```

如果你的服务器需要指定 Python：

```bash
PYTHON_BIN=python3.11 bash scripts/server/00_setup_env.sh
source .venv/bin/activate
```

如果服务器的 PyTorch CUDA 需要特殊安装命令，可以先手动安装对应版本的 torch，再执行：

```bash
python -m pip install -r requirements-model.txt
```

## 4. 一键跑完整流程

默认配置：

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- local model dir: `models/qwen2.5-0.5b-instruct`
- SFT steps: `200`
- rollout: `100 tasks x group size 4`
- rollout split: `tasks/train_tasks.jsonl`
- GRPO-KL steps: `100`
- KL beta: `0.02`
- eval: `200 tasks`

运行：

```bash
source .venv/bin/activate
bash scripts/server/run_grpo_kl_pipeline.sh
```

## 5. 分步运行

如果你希望更稳地观察每一步，可以分开跑。

下载模型：

```bash
bash scripts/server/01_download_model.sh
```

训练 SFT adapter：

```bash
bash scripts/server/02_train_sft.sh
```

采样 GRPO rollouts：

```bash
bash scripts/server/03_collect_grpo_rollouts.sh
```

训练 GRPO-KL adapter：

```bash
bash scripts/server/04_train_grpo_kl.sh
```

评测并生成对比：

```bash
bash scripts/server/05_eval_compare.sh
```

## 6. 调整实验规模

更保守：

```bash
ROLLOUT_LIMIT=50 GROUP_SIZE=4 GRPO_STEPS=50 EVAL_LIMIT=100 bash scripts/server/run_grpo_kl_pipeline.sh
```

更有价值的正式一点的实验：

```bash
ROLLOUT_LIMIT=100 GROUP_SIZE=4 GRPO_STEPS=100 EVAL_LIMIT=200 bash scripts/server/run_grpo_kl_pipeline.sh
```

更激进：

```bash
ROLLOUT_LIMIT=100 GROUP_SIZE=8 GRPO_STEPS=200 EVAL_LIMIT=200 bash scripts/server/run_grpo_kl_pipeline.sh
```

如果 `GROUP_SIZE=8` 太慢，退回 `GROUP_SIZE=4`。

## 7. 关键输出

SFT adapter：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200
```

GRPO rollout：

```text
outputs/rollouts/grpo_sft_step200_train_group4_task100.jsonl
outputs/eval_reports/grpo_sft_step200_train_group4_task100_report.json
```

GRPO-KL adapter：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_grpo_kl_task100_group4_step100
```

SFT eval：

```text
outputs/eval_reports/server_sft_eval200_report.json
outputs/trajectories/server_sft_eval200_trajectories.jsonl
```

GRPO-KL eval：

```text
outputs/eval_reports/server_grpo_kl_eval200_report.json
outputs/trajectories/server_grpo_kl_eval200_trajectories.jsonl
```

对比报告：

```text
outputs/eval_reports/server_sft_vs_grpo_kl_comparison.json
outputs/eval_reports/server_sft_vs_grpo_kl_comparison.md
```

## 8. 断点续跑

rollout 和 eval 脚本默认使用：

```text
--incremental --resume
```

所以中断后重新执行同一条命令，会跳过已经完成的 task。

注意：如果你修改了 `ROLLOUT_LIMIT`、`GROUP_SIZE` 或输出路径，最好确认对应输出文件名也变了，避免把不同配置的数据混在一起。

## 9. 先看哪些指标

第一优先级：

- `task_success_rate`
- `tool_call_format_accuracy`
- `invalid_tool_call_rate`

第二优先级：

- `average_model_steps`
- failures 数量
- error analysis 里的 `wrong_click_path` / `wrong_candidate_after_filter`

判断标准：

- 如果 GRPO-KL success rate 高于 SFT，同时 format/invalid 不退化，这是正向结果。
- 如果 success 持平但 invalid 降低，也值得记录。
- 如果 success 下降但 KL/loss 正常，说明 reward 或 rollout 数据还不够，需要调数据和超参。

训练 rollout 必须来自 `tasks/train_tasks.jsonl`，最终指标必须来自独立的
`tasks/eval_tasks.jsonl`。不要在 eval tasks 上采样并训练后，再把同一批任务作为最终评测。

## 10. 常见问题

### Hugging Face 下载失败

先设置：

```bash
export HF_HUB_DISABLE_XET=1
```

如果是国内网络，可以考虑提前在本地下载或使用镜像，但要保证模型目录里有 tokenizer/config/safetensors 文件。

### CUDA 不可用

检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果是 `False`，先处理服务器 CUDA/PyTorch 安装，不要直接用 CPU 跑完整 rollout。

### SFT adapter 已经上传

如果你已经上传了：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200
```

可以跳过：

```bash
bash scripts/server/02_train_sft.sh
```

直接从 rollout 开始。

## 11. 面试里怎么解释服务器实验

可以这样说：

> 本地机器已经完成了环境、SFT、reward 和 GRPO-KL 的功能验证。服务器实验主要是为了扩大 rollout 数量，因为 RL 阶段最耗时的是同一任务的多样本采样。服务器上我会固定同一套 eval split，对比 SFT adapter 和 GRPO-KL adapter 的成功率、格式准确率和非法工具率，这样能判断 RL 是否真的带来收益，而不是只看训练 loss。
