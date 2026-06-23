# Step 01: Base Model Eval Readiness

本步骤的目标是把真实 Base model 评估前的基础设施准备好。

这里的“完成”不表示已经跑出了某个真实大模型的 Base 分数，而是表示评估框架已经可以接收一个本地 Hugging Face causal LM，让它在 WebNav 环境里逐步输出 tool call、执行工具、记录轨迹，并产出统一指标报告。

## 1. 这一步解决什么问题

在 SFT 之前，我们需要一个 Base model baseline。否则后面即使 SFT 或 GRPO 跑出结果，也无法说明提升来自哪里。

Base eval 要回答这些问题：

- Base model 能不能稳定输出 `<tool_call>{...}</tool_call>` 格式？
- 它会不会输出自然语言、多个 tool call、错误 JSON？
- 它会不会调用不存在的工具？
- 它会不会点击不存在的 `element_id`？
- 它最终能不能 `submit_answer` 并答对？
- 它平均需要多少步？

所以这一步的核心不是训练，而是让评估路径先闭环：

```text
task
-> prompt messages
-> model output
-> tool-call parser
-> tool registry
-> BrowserEnv step
-> verifier summary
-> metrics report
```

## 2. 当前新增/确认的模块

### 2.1 Tool-call parser

文件：

```text
rollout/parser.py
```

作用：

解析模型输出，要求模型必须输出严格的一段：

```text
<tool_call>{"name": "click", "arguments": {"element_id": "xxx"}}</tool_call>
```

它会拒绝几类错误：

- 空输出：`empty_output`
- 没有完整 wrapper：`invalid_wrapper`
- JSON 解析失败：`invalid_json`
- schema 不符合要求：`invalid_schema`

为什么重要：

> Base model 常见问题不是“答错”，而是根本没有按工具协议输出。parser 把这些错误显式记录下来，后续可以计算 tool call format accuracy。

### 2.2 Model runner

文件：

```text
rollout/model_runner.py
```

作用：

把一个 `TextGenerator` 接到环境里执行。它不关心 generator 是真实模型、mock 模型还是 expert replay，只要求 generator 接收 messages 并返回一段文本。

关键流程：

1. 初始化 `BrowserEnv`，但不自动打开页面。
2. 给模型 system prompt 和 user instruction。
3. 每一步调用 generator 生成文本。
4. 用 parser 解析 tool call。
5. 通过 `ToolRegistry` 执行工具。
6. 把 tool observation 追加回 messages。
7. 如果调用了 `submit_answer`，episode 结束。
8. 输出 trajectory 和 summary。

这让真实模型评估和 expert replay smoke test 走同一条路径。

### 2.3 Tool registry

文件：

```text
tools/tool_registry.py
```

作用：

根据 tool name 分发到具体工具：

```text
open_page
click
get_visible_text
submit_answer
```

它现在会处理两类 invalid action：

- 工具名不存在。
- 参数不匹配，比如 `click` 少传 `element_id`。

这些错误会计入 `invalid_tool_calls`。

### 2.4 Transformers generator

文件：

```text
rollout/transformers_generator.py
```

作用：

把本地 Hugging Face 模型封装成 `TextGenerator`。

它使用：

```python
AutoTokenizer.from_pretrained(...)
AutoModelForCausalLM.from_pretrained(...)
tokenizer.apply_chat_template(...)
model.generate(...)
```

这意味着后续可以这样接模型：

```bash
python scripts/run_eval.py --model path_or_hf_model_id
```

如果本地没有安装 `torch` 和 `transformers`，会提示安装：

```text
requirements-model.txt
```

### 2.5 Metrics

文件：

```text
eval/metrics.py
```

当前指标：

```text
task_success_rate
final_answer_accuracy
tool_call_format_accuracy
invalid_tool_call_rate
average_model_steps
average_environment_actions
format_errors
invalid_tool_calls
termination_counts
```

指标解释：

- `task_success_rate`：最终提交答案是否正确。
- `final_answer_accuracy`：当前等同于 success rate，后续可和路径成功拆开。
- `tool_call_format_accuracy`：模型输出能被 parser 正确解析的比例。
- `invalid_tool_call_rate`：格式错误、未知工具、参数错误、环境执行错误占所有模型步骤的比例。
- `average_model_steps`：模型平均生成几步。
- `average_environment_actions`：环境记录了几次工具动作。
- `termination_counts`：episode 是正常 submitted，还是 max_steps 截断。

### 2.6 Eval entrypoint

文件：

```text
scripts/run_eval.py
```

两种模式：

1. 不传 `--model`：使用 expert replay，作为评估管线 smoke test。
2. 传 `--model`：使用真实 Hugging Face 模型。

## 3. 本步骤做的一个小增强

之前 eval report 只包含指标，没有说明这次评估是 expert replay 还是真实模型。

这容易造成误读，比如看到：

```json
"task_success_rate": 1.0
```

但不知道它其实是 expert replay 的 smoke test，而不是 Base model 成绩。

所以本步骤给 `evaluate_tasks` 和 `run_eval.py` 增加了 `metadata`：

```json
{
  "metadata": {
    "eval_mode": "expert_replay",
    "model": null,
    "tasks": "tasks/eval_tasks.jsonl",
    "limit": 10,
    "device": null,
    "max_new_tokens": null,
    "temperature": null,
    "trust_remote_code": null
  }
}
```

如果跑真实模型，会变成：

```json
{
  "metadata": {
    "eval_mode": "transformers_model",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "tasks": "tasks/eval_tasks.jsonl",
    "limit": 200,
    "device": "auto",
    "max_new_tokens": 256,
    "temperature": 0.0,
    "trust_remote_code": false
  }
}
```

这个增强看起来小，但对实验管理很重要。后续 Base/SFT/GRPO 会产生很多 report，如果没有 metadata，很容易把 smoke test、base eval、sft eval 混在一起。

## 4. 已验证的命令

### 4.1 单元测试

```bash
python -m pytest -q
```

当前结果：

```text
8 passed
```

覆盖内容：

- tool call parser 能解析合法输出。
- parser 能拒绝自然语言包裹、非法 JSON、非法 schema。
- expert replay 能完成任务。
- parser error 会被计数。
- metrics 聚合正确。
- unknown tool / bad arguments 会计入 invalid action。

### 4.2 Expert replay smoke eval

```bash
python scripts/run_eval.py \
  --limit 10 \
  --output outputs/trajectories/eval_smoke_trajectories.jsonl \
  --report outputs/eval_reports/eval_smoke_report.json \
  --failures outputs/eval_reports/eval_smoke_failures.jsonl
```

当前结果：

```json
{
  "num_tasks": 10,
  "successes": 10,
  "task_success_rate": 1.0,
  "final_answer_accuracy": 1.0,
  "tool_call_format_accuracy": 1.0,
  "invalid_tool_call_rate": 0.0,
  "average_model_steps": 3.5,
  "average_environment_actions": 3.5,
  "format_errors": 0,
  "invalid_tool_calls": 0,
  "termination_counts": {
    "submitted": 10
  },
  "metadata": {
    "eval_mode": "expert_replay",
    "model": null,
    "tasks": "tasks/eval_tasks.jsonl",
    "limit": 10,
    "device": null,
    "max_new_tokens": null,
    "temperature": null,
    "trust_remote_code": null
  }
}
```

注意：

> 这个结果不是 Base model 能力，而是 expert replay 的管线验证。它证明 parser、tool registry、BrowserEnv、verifier、metrics、report 写出都能正常工作。

## 5. 如何跑真实 Base model

如果已经有本地模型或能访问 Hugging Face 模型，可以跑：

```bash
python scripts/run_eval.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks tasks/eval_tasks.jsonl \
  --limit 200 \
  --output outputs/trajectories/base_eval_trajectories.jsonl \
  --report outputs/eval_reports/base_eval_report.json \
  --failures outputs/eval_reports/base_eval_failures.jsonl \
  --device auto \
  --max-new-tokens 256 \
  --temperature 0.0
```

如果模型已经下载到本地，则把 `--model` 改成本地路径，例如：

```bash
python scripts/run_eval.py \
  --model D:\path\to\Qwen2.5-0.5B-Instruct \
  --limit 200 \
  --output outputs/trajectories/base_eval_trajectories.jsonl \
  --report outputs/eval_reports/base_eval_report.json \
  --failures outputs/eval_reports/base_eval_failures.jsonl
```

如果缺依赖：

```bash
pip install -r requirements-model.txt
```

## 6. 面试时怎么讲这一步

可以这样说：

> 我在 SFT 前先实现了模型 rollout/eval 框架。模型每一步只能输出一个 `<tool_call>{...}</tool_call>`，parser 会严格检查 wrapper、JSON 和 schema。合法 tool call 会通过 ToolRegistry 分发到 BrowserEnv，环境返回 observation，再进入下一轮 messages。最后 verifier 判断答案是否正确，metrics 统计 success rate、format accuracy、invalid tool call rate 和 average steps。为了避免混淆，我还在 eval report 里记录 metadata，区分 expert replay smoke test 和真实模型评估。

如果被问“现在的 100% 成功是不是模型成绩”，要明确回答：

> 不是。当前 100% 是 expert replay 的 smoke test，用来证明评估管线正确。真实 Base model 需要通过 `--model` 接入本地或 Hugging Face 模型后单独生成 `base_eval_report.json`。

## 7. 当前状态和下一步

当前状态：

```text
Base model eval framework: ready
Expert replay smoke test: passed
Unit tests: passed
Real Base model score: not run yet
```

下一步：

1. 确认本地模型路径或安装模型依赖。
2. 跑真实 Base model eval。
3. 保存 `base_eval_report.json` 和失败案例。
4. 基于 `training/sft_train.jsonl` 写 LoRA SFT 脚本。
5. 训练 SFT 后再跑同一套 eval，对比 Base vs SFT。
