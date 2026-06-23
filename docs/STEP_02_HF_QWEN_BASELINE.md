# Step 02: Hugging Face Qwen Base Baseline

本步骤完成了第一轮真实 Hugging Face 模型接入，并跑出了 Qwen2.5-0.5B-Instruct 的小规模 Base baseline。

这一步和上一份 `STEP_01_BASE_EVAL_READINESS.md` 的区别是：

- Step 01 验证的是评估框架，用的是 expert replay，不是真模型。
- Step 02 真的下载并加载了 Hugging Face 模型，让模型自己生成输出。

## 1. 使用的模型

模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

本地保存路径：

```text
models/qwen2.5-0.5b-instruct
```

选择它的原因：

- 模型足够小，适合 12GB 显存笔记本。
- 是 instruct 模型，具备基础指令跟随能力。
- 后续也适合作为 LoRA SFT 的起点。

## 2. 本机环境

使用的 Python 环境：

```text
D:\Program\Anaconda\envs\research\python.exe
```

GPU：

```text
NVIDIA GeForce RTX 5070 Laptop
显存约 12GB
```

这个环境里原本有 `torch` 和 `transformers`，但缺少一些运行依赖。为了不改坏 Anaconda 环境，我把少量缺失依赖安装到了项目内：

```text
.python_deps/research
```

运行时通过临时 `PYTHONPATH` 加载：

```powershell
$env:PYTHONPATH='D:\job\Program\WebNav-RL\.python_deps\research'
```

补过的关键依赖包括：

```text
typing_extensions
numpy
filelock
sympy
jinja2
```

注意：`.python_deps/` 和 `models/` 已经加入 `.gitignore`，不应该提交到仓库。

## 3. Hugging Face 下载过程

直接运行模型评估时，第一次下载因为沙箱网络限制失败。授权联网后，下载仍多次被远端中断。

最终采用的稳定方式是：

```powershell
$env:PYTHONPATH='D:\job\Program\WebNav-RL\.python_deps\research'
$env:HF_HUB_DISABLE_XET='1'
D:\Program\Anaconda\envs\research\python.exe -c "from huggingface_hub import snapshot_download; path=snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/qwen2.5-0.5b-instruct', allow_patterns=['*.json','*.jinja','*.txt','*.safetensors'], max_workers=1); print(path)"
```

关键点：

- `HF_HUB_DISABLE_XET=1`：禁用 Xet 下载路径，使用普通 HTTP 下载。
- `max_workers=1`：单线程下载，减少连接波动。
- 多次重跑可以断点续传。

下载过程里权重文件大约 988MB，中间断过几次，但最终完整下载成功。

## 4. 为什么加了 `--max-steps`

真实 Base model 第一次跑 3 条完整 rollout 时，模型完全没有输出工具调用，而是直接输出自然语言。比如：

```text
The product that costs 99 dollars is likely a calculator or a similar device priced at $99.
```

这不符合环境要求的格式：

```text
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
```

如果每条任务都让它继续跑满 8 步，它会不断收到 parser error，然后继续自然语言解释，速度很慢，信息增量也不大。

所以本步骤给 `scripts/run_eval.py` 增加了：

```text
--max-steps
```

相关修改：

```text
eval/evaluate.py
scripts/run_eval.py
```

这样可以先做一个快速 baseline：

- Base model 第一轮是否能输出合法 tool call？
- 如果第一轮已经 0%，说明 SFT 的首要目标非常明确：学习工具调用协议。

## 5. 已运行的评估

### 5.1 3 条完整 rollout smoke test

命令：

```powershell
$env:PYTHONPATH='D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --tasks tasks\eval_tasks.jsonl `
  --limit 3 `
  --output outputs\trajectories\base_qwen_0_5b_smoke_trajectories.jsonl `
  --report outputs\eval_reports\base_qwen_0_5b_smoke_report.json `
  --failures outputs\eval_reports\base_qwen_0_5b_smoke_failures.jsonl `
  --device auto `
  --max-new-tokens 128 `
  --temperature 0.0
```

结果：

```json
{
  "num_tasks": 3,
  "successes": 0,
  "task_success_rate": 0.0,
  "final_answer_accuracy": 0.0,
  "tool_call_format_accuracy": 0.0,
  "invalid_tool_call_rate": 1.0,
  "average_model_steps": 8.0,
  "average_environment_actions": 0.0,
  "format_errors": 24,
  "invalid_tool_calls": 24,
  "termination_counts": {
    "max_steps": 3
  }
}
```

解释：

- 3 条任务都没有成功。
- 每条任务跑满 8 步。
- 24 次模型输出全部是格式错误。
- 环境没有执行任何有效 action，因为 parser 从未得到合法 tool call。

### 5.2 20 条第一步格式 baseline

命令：

```powershell
$env:PYTHONPATH='D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --tasks tasks\eval_tasks.jsonl `
  --limit 20 `
  --max-steps 1 `
  --output outputs\trajectories\base_qwen_0_5b_eval20_step1_trajectories.jsonl `
  --report outputs\eval_reports\base_qwen_0_5b_eval20_step1_report.json `
  --failures outputs\eval_reports\base_qwen_0_5b_eval20_step1_failures.jsonl `
  --device auto `
  --max-new-tokens 64 `
  --temperature 0.0
```

结果：

```json
{
  "num_tasks": 20,
  "successes": 0,
  "task_success_rate": 0.0,
  "final_answer_accuracy": 0.0,
  "tool_call_format_accuracy": 0.0,
  "invalid_tool_call_rate": 1.0,
  "average_model_steps": 1.0,
  "average_environment_actions": 0.0,
  "format_errors": 20,
  "invalid_tool_calls": 20,
  "termination_counts": {
    "max_steps": 20
  }
}
```

报告文件：

```text
outputs/eval_reports/base_qwen_0_5b_eval20_step1_report.json
```

失败案例：

```text
outputs/eval_reports/base_qwen_0_5b_eval20_step1_failures.jsonl
```

完整轨迹：

```text
outputs/trajectories/base_qwen_0_5b_eval20_step1_trajectories.jsonl
```

## 6. 这个结果说明什么

这个 baseline 很有价值，虽然指标是 0。

它说明：

1. 真实 Hugging Face 模型已经接入成功。
2. 模型能加载到 GPU/本机环境并生成文本。
3. 当前 Base model 不会自然遵守项目定义的 tool-call protocol。
4. SFT 的目标非常明确：先把 tool call 格式学稳定。

换句话说，后续如果 SFT 后 `tool_call_format_accuracy` 从 0 提升到一个明显高的数，这就是一个清晰的实验结果。

## 7. 面试时怎么讲

可以这样讲：

> 我接入了 Hugging Face 的 Qwen2.5-0.5B-Instruct 作为 Base model baseline。模型能正常加载和生成，但在当前 WebNav 环境里，Base model 并不会天然输出我定义的 `<tool_call>{...}</tool_call>` 协议，而是倾向于直接用自然语言回答。20 条 eval task 的第一步格式评估里，tool call format accuracy 是 0，invalid tool call rate 是 1。这说明 SFT 阶段的首要任务不是直接提升最终成功率，而是先教会小模型稳定遵守工具调用协议。

如果被问“为什么只跑 20 条 / 为什么 max steps 是 1”，可以回答：

> 我先跑了 3 条完整 8-step rollout，发现每一步都是 parser error。继续扩大完整 rollout 只会重复同一种错误，所以我加了 `--max-steps 1`，先快速测 Base model 的第一步工具格式能力。等 SFT 后模型能输出合法 tool call，再跑完整 200 条 eval 测任务成功率。

## 8. 当前状态

```text
Hugging Face Qwen model downloaded: yes
Local model path: models/qwen2.5-0.5b-instruct
3-task full rollout smoke test: completed
20-task step-1 Base format baseline: completed
Base tool call format accuracy: 0.0
Base invalid tool call rate: 1.0
```

## 9. 下一步

下一步应该进入 SFT 准备：

1. 写 LoRA SFT 脚本。
2. 使用 `training/sft_train.jsonl` 训练 Qwen2.5-0.5B-Instruct。
3. 保存 SFT adapter。
4. 用同一套 eval 跑 SFT model。
5. 先比较第一步 tool-call format accuracy，再比较完整 rollout success rate。

预期实验主线：

```text
Base: tool-call format accuracy = 0.0
SFT: tool-call format accuracy 明显提升
SFT complete rollout: 开始出现有效 open_page/click/submit_answer
```
