# Step 03: LoRA SFT for Tool-Call Protocol

本步骤完成了第一版 LoRA SFT：用 expert trajectories 训练 Qwen2.5-0.5B-Instruct，让模型学习 WebNav 环境要求的工具调用格式和基本多步行为。

这一步已经产生了一个非常清晰的阶段性结果：

```text
Base first-step tool_call_format_accuracy: 0.0
SFT first-step tool_call_format_accuracy: 1.0
```

也就是说，SFT 已经把模型从“完全不会输出工具调用协议”拉到了“能稳定输出合法 `<tool_call>`”。

## 1. 为什么这一步先做 SFT

上一阶段的真实 Base baseline 显示，Qwen2.5-0.5B-Instruct 在我们的 WebNav 环境里会直接输出自然语言。例如它会回答：

```text
The product that costs 99 dollars is likely a calculator...
```

但环境需要的是严格工具调用：

```text
<tool_call>{"name": "open_page", "arguments": {"page_id": "shop_home"}}</tool_call>
```

所以当前 SFT 的核心目标不是马上追求高任务成功率，而是先让小模型学会：

```text
instruction -> tool call
tool observation -> next tool call
detail page -> submit_answer
```

这一步是后续 GRPO 的前置条件。否则 RL rollout 会大量浪费在 parser error 上，reward 信号也会很稀疏。

## 2. 新增脚本

文件：

```text
training/sft_train.py
```

功能：

- 读取 `training/sft_train.jsonl`。
- 把每个 assistant tool call 拆成一个 next-action 训练样本。
- 用 Qwen chat template 构造 prompt。
- 只对 assistant 的目标 tool call 计算 loss。
- 用 PEFT LoRA 注入少量可训练参数。
- 保存 LoRA adapter 和训练 metadata。

## 3. 为什么按 next-action 训练

原始 SFT 数据是一整条多轮 trajectory：

```text
user instruction
assistant open_page
tool observation
assistant click
tool observation
assistant submit_answer
```

如果直接把整条 conversation 当成一个长样本训练，模型也能学，但不够贴近实际 rollout。

实际 rollout 时，模型每次只需要生成“下一步 action”：

```text
已有 messages -> assistant 下一条 <tool_call>
```

所以脚本会把一条 trajectory 拆成多个训练样本：

```text
sample 1: system + user -> open_page
sample 2: system + user + open_page + observation -> click
sample 3: ... -> submit_answer
```

这样训练目标和推理时的行为完全一致。

## 4. LoRA 配置

本次使用的默认 LoRA 配置：

```text
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules:
  q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj
```

训练时打印的可训练参数：

```text
trainable params: 4,399,104
all params: 498,431,872
trainable%: 0.8826
```

这说明只训练不到 1% 的参数，适合本机 12GB 显存。

## 5. 训练命令

本次先做小规模 SFT，不直接跑完整 800 条。

命令：

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe training\sft_train.py `
  --model models\qwen2.5-0.5b-instruct `
  --train-data training\sft_train.jsonl `
  --output-dir outputs\checkpoints\qwen2_5_0_5b_lora_sft_step40 `
  --limit-rows 200 `
  --max-steps 40 `
  --max-seq-len 768 `
  --batch-size 1 `
  --gradient-accumulation-steps 4 `
  --learning-rate 2e-4 `
  --log-every 5
```

输出 adapter：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step40
```

训练 metadata：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step40/training_metadata.json
```

本次训练设置：

```text
rows: 200 trajectories
next_action_examples: 631
optimizer_steps: 40
batch_size: 1
gradient_accumulation_steps: 4
max_seq_len: 768
learning_rate: 2e-4
```

训练 loss 从第一步的约 `1.39` 降到了后期的 `0.08` 左右。这个 loss 只是小规模训练参考，不代表最终模型质量，但说明模型确实在快速学习工具调用样式。

## 6. Eval 脚本支持 LoRA adapter

为了评估 SFT 模型，本步骤也扩展了：

```text
rollout/transformers_generator.py
scripts/run_eval.py
```

新增参数：

```text
--adapter
```

现在可以用：

```powershell
python scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step40
```

这样加载的是：

```text
base model + LoRA adapter
```

报告 metadata 也会记录 adapter 路径，避免和 Base eval 混淆。

## 7. 第一项结果：20 条第一步格式评估

命令：

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step40 `
  --tasks tasks\eval_tasks.jsonl `
  --limit 20 `
  --max-steps 1 `
  --output outputs\trajectories\sft_qwen_0_5b_step40_eval20_step1_trajectories.jsonl `
  --report outputs\eval_reports\sft_qwen_0_5b_step40_eval20_step1_report.json `
  --failures outputs\eval_reports\sft_qwen_0_5b_step40_eval20_step1_failures.jsonl `
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
  "tool_call_format_accuracy": 1.0,
  "invalid_tool_call_rate": 0.0,
  "average_model_steps": 1.0,
  "average_environment_actions": 1.0,
  "format_errors": 0,
  "invalid_tool_calls": 0
}
```

对比 Base：

| Model | Eval | Tool Call Format Accuracy | Invalid Tool Call Rate |
|---|---|---:|---:|
| Base Qwen2.5-0.5B-Instruct | 20 tasks, 1 step | 0.0 | 1.0 |
| SFT step40 LoRA | 20 tasks, 1 step | 1.0 | 0.0 |

这个结果非常关键。它证明 SFT 已经解决了当前项目的第一个核心问题：让小模型遵守工具调用协议。

抽样输出：

```text
<tool_call>{"name": "open_page", "arguments": {"page_id": "shop_home"}}</tool_call>
```

## 8. 第二项结果：20 条完整 rollout

命令：

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step40 `
  --tasks tasks\eval_tasks.jsonl `
  --limit 20 `
  --output outputs\trajectories\sft_qwen_0_5b_step40_eval20_full_trajectories.jsonl `
  --report outputs\eval_reports\sft_qwen_0_5b_step40_eval20_full_report.json `
  --failures outputs\eval_reports\sft_qwen_0_5b_step40_eval20_full_failures.jsonl `
  --device auto `
  --max-new-tokens 96 `
  --temperature 0.0
```

结果：

```json
{
  "num_tasks": 20,
  "successes": 3,
  "task_success_rate": 0.15,
  "final_answer_accuracy": 0.15,
  "tool_call_format_accuracy": 1.0,
  "invalid_tool_call_rate": 0.0,
  "average_model_steps": 3.0,
  "average_environment_actions": 3.0,
  "termination_counts": {
    "submitted": 20
  }
}
```

解释：

- 模型已经能稳定输出合法工具调用。
- 模型每条任务都能走到 `submit_answer`。
- 20 条中答对 3 条，成功率 15%。
- 没有 parser error，也没有 invalid tool call。

这个完整 rollout 结果说明：小规模 SFT 不只学到了 wrapper 格式，也初步学到了 open/click/submit 的行为模式。但它的答案选择能力还弱，需要更充分训练和可能更好的数据构造。

## 9. 面试时怎么讲

可以这样讲：

> Base Qwen2.5-0.5B-Instruct 在这个环境里不会自然输出工具调用，而是直接自然语言回答。于是我用 expert trajectories 做 LoRA SFT。训练时我没有简单把整条 trajectory 拼成一个样本，而是把每个 assistant tool call 拆成 next-action prediction，让训练目标和 rollout 时一致。小规模训练 40 个 optimizer steps 后，20 条 eval 的第一步 tool-call format accuracy 从 0 提升到 1.0，invalid tool call rate 从 1.0 降到 0。完整 rollout 上也能稳定 submit，20 条里成功 3 条。这证明 SFT 已经把模型带进了可交互的工具调用空间。

如果被问为什么完整任务成功率还不高：

> 当前只是 200 条轨迹、40 steps 的小规模 SFT，主要目标是验证 tool-call protocol learning。任务成功率需要更充分训练、更完整的 800 条数据，以及后续可能的 reward-based fine-tuning。现在最重要的阶段性结果是 parser error 已经从大量出现变成 0。

## 10. 当前局限

这一步还有几个局限：

- 只训练了前 200 条 trajectories。
- 只跑了 40 个 optimizer steps。
- eval 只跑了 20 条，不是完整 200 条。
- 完整 rollout 成功率只有 15%。
- 当前没有做 validation loss 或 checkpoint selection。
- 当前没有 error analysis，尚未分类失败原因。

这些局限是后续工作，不影响这一步的结论：SFT 已经显著提升工具调用格式稳定性。

## 11. 下一步

下一步建议：

1. 跑更完整的 SFT：
   ```text
   rows: 800
   max_steps: 150-300
   max_seq_len: 1024
   ```
2. 跑完整 200 条 SFT eval。
3. 写 error analysis，看失败是选错 element、过早 submit，还是答案提取错误。
4. 如果 SFT 稳定后，再进入 reward function 和 GRPO。

推荐的下一条训练命令可以从当前命令扩展：

```powershell
D:\Program\Anaconda\envs\research\python.exe training\sft_train.py `
  --model models\qwen2.5-0.5b-instruct `
  --train-data training\sft_train.jsonl `
  --output-dir outputs\checkpoints\qwen2_5_0_5b_lora_sft_fuller `
  --limit-rows 800 `
  --max-steps 200 `
  --max-seq-len 1024 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --learning-rate 2e-4
```
