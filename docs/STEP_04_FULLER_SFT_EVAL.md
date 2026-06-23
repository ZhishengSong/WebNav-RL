# Step 04: Fuller SFT Run and Evaluation

本步骤把上一轮小规模 SFT 扩大到完整 800 条训练轨迹，并跑出更有说服力的 SFT eval 结果。

核心结论：

```text
Base first-step format accuracy: 0.0
SFT step40 full rollout success rate: 0.15 on 20 tasks
SFT step200 full rollout success rate: 0.64 on 50 tasks
SFT step200 first-step format accuracy: 1.0 on 200 tasks
```

这说明更完整的 SFT 不只学会了 tool-call wrapper，还显著提升了完整任务成功率。

## 1. 本步骤目标

上一阶段的 `step40` adapter 已经证明：

- Base model 不会输出工具调用格式。
- 小规模 SFT 后，模型能稳定输出合法 `<tool_call>`。
- 20 条完整 rollout 成功率达到 15%。

但 `step40` 只用了：

```text
200 trajectories
631 next-action examples
40 optimizer steps
```

所以本步骤扩大训练：

```text
800 trajectories
2530 next-action examples
200 optimizer steps
```

目标是看：

- 格式准确率能否在完整 200 条 eval 上保持 100%。
- 完整 rollout 的任务成功率是否明显提升。
- 本机 12GB 显存能否支撑更完整的 LoRA SFT。

## 2. 训练命令

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe training\sft_train.py `
  --model models\qwen2.5-0.5b-instruct `
  --train-data training\sft_train.jsonl `
  --output-dir outputs\checkpoints\qwen2_5_0_5b_lora_sft_step200 `
  --limit-rows 800 `
  --max-steps 200 `
  --max-seq-len 1024 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --learning-rate 2e-4 `
  --log-every 10
```

输出 adapter：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200
```

训练 metadata：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200/training_metadata.json
```

训练规模：

```text
rows: 800
next_action_examples: 2530
optimizer_steps: 200
batch_size: 1
gradient_accumulation_steps: 8
max_seq_len: 1024
learning_rate: 2e-4
```

LoRA 参数：

```text
trainable params: 4,399,104
all params: 498,431,872
trainable%: 0.8826
```

训练时间：

```text
约 5 分钟
```

这说明当前阶段继续用本机是可行的，不需要服务器。

## 3. 训练 loss

训练日志显示 loss 很快下降：

```text
step 1:   1.7896
step 10:  0.0368
step 50:  0.0799
step 100: 0.0210
step 150: 0.0158
step 200: 0.0234
```

这个 loss 说明模型非常快地学会了 expert trajectory 的 tool-call 分布。

注意：loss 低不等于任务成功率高。任务成功率还取决于模型在 rollout 中是否能根据 observation 选择正确 item 和正确 answer。

## 4. Eval 1：200 条第一步格式评估

命令：

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step200 `
  --tasks tasks\eval_tasks.jsonl `
  --limit 200 `
  --max-steps 1 `
  --output outputs\trajectories\sft_qwen_0_5b_step200_eval200_step1_trajectories.jsonl `
  --report outputs\eval_reports\sft_qwen_0_5b_step200_eval200_step1_report.json `
  --failures outputs\eval_reports\sft_qwen_0_5b_step200_eval200_step1_failures.jsonl `
  --device auto `
  --max-new-tokens 64 `
  --temperature 0.0
```

结果：

```json
{
  "num_tasks": 200,
  "task_success_rate": 0.0,
  "tool_call_format_accuracy": 1.0,
  "invalid_tool_call_rate": 0.0,
  "average_model_steps": 1.0,
  "format_errors": 0,
  "invalid_tool_calls": 0
}
```

解释：

- 这是只跑第一步的格式评估，不测试最终成功率。
- 200 条 eval 里，第一步工具调用格式 100% 合法。
- Base model 在同类评估中是 0%。

这一项是 SFT 最关键的协议学习结果。

## 5. Eval 2：20 条完整 rollout

一开始我尝试跑 200 条完整 rollout，使用 `max-new-tokens=96`，但超过 40 分钟没有结束。随后尝试 50 条同样配置，也超过 30 分钟。

原因判断：

- 完整 rollout 每条任务可能需要多步生成。
- `max-new-tokens=96` 对 tool call 来说偏大。
- 如果某些任务生成慢，长评估没有增量保存，超时后没有报告。

因此我先把 `max-new-tokens` 降到 48，跑 20 条完整 rollout。

命令：

```powershell
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step200 `
  --tasks tasks\eval_tasks.jsonl `
  --limit 20 `
  --output outputs\trajectories\sft_qwen_0_5b_step200_eval20_full_trajectories.jsonl `
  --report outputs\eval_reports\sft_qwen_0_5b_step200_eval20_full_report.json `
  --failures outputs\eval_reports\sft_qwen_0_5b_step200_eval20_full_failures.jsonl `
  --device auto `
  --max-new-tokens 48 `
  --temperature 0.0
```

结果：

```json
{
  "num_tasks": 20,
  "successes": 14,
  "task_success_rate": 0.7,
  "final_answer_accuracy": 0.7,
  "tool_call_format_accuracy": 1.0,
  "invalid_tool_call_rate": 0.0303,
  "average_model_steps": 3.3,
  "termination_counts": {
    "submitted": 20
  }
}
```

## 6. Eval 3：50 条完整 rollout

为了得到更稳的结果，又跑了 50 条完整 rollout，继续使用 `max-new-tokens=48`。

命令：

```powershell
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step200 `
  --tasks tasks\eval_tasks.jsonl `
  --limit 50 `
  --output outputs\trajectories\sft_qwen_0_5b_step200_eval50_full_t48_trajectories.jsonl `
  --report outputs\eval_reports\sft_qwen_0_5b_step200_eval50_full_t48_report.json `
  --failures outputs\eval_reports\sft_qwen_0_5b_step200_eval50_full_t48_failures.jsonl `
  --device auto `
  --max-new-tokens 48 `
  --temperature 0.0
```

结果：

```json
{
  "num_tasks": 50,
  "successes": 32,
  "task_success_rate": 0.64,
  "final_answer_accuracy": 0.64,
  "tool_call_format_accuracy": 1.0,
  "invalid_tool_call_rate": 0.0185,
  "average_model_steps": 3.24,
  "average_environment_actions": 3.24,
  "format_errors": 0,
  "invalid_tool_calls": 3,
  "termination_counts": {
    "submitted": 50
  }
}
```

报告文件：

```text
outputs/eval_reports/sft_qwen_0_5b_step200_eval50_full_t48_report.json
```

失败案例：

```text
outputs/eval_reports/sft_qwen_0_5b_step200_eval50_full_t48_failures.jsonl
```

轨迹：

```text
outputs/trajectories/sft_qwen_0_5b_step200_eval50_full_t48_trajectories.jsonl
```

## 7. 对比总结

| Model | Eval setting | Success Rate | Format Accuracy | Invalid Tool Call Rate |
|---|---|---:|---:|---:|
| Base Qwen2.5-0.5B | 20 tasks, first step | 0.00 | 0.00 | 1.00 |
| SFT step40 | 20 tasks, full rollout | 0.15 | 1.00 | 0.00 |
| SFT step200 | 200 tasks, first step | 0.00 | 1.00 | 0.00 |
| SFT step200 | 50 tasks, full rollout | 0.64 | 1.00 | 0.0185 |

最重要的结论：

- SFT 已经稳定解决 tool-call protocol。
- 更完整训练把 full rollout success 从 15% 提升到 64%。
- 模型平均 3.24 步完成任务，说明它基本学到了 `open_page -> click -> submit_answer` 的流程。
- invalid tool call 仍然很低，只有 3 次。

## 8. 为什么没有直接报告 200 条完整 rollout

这一步尝试过 200 条完整 rollout，但长时间没有完成，而且当前 `evaluate_tasks` 是评估全部结束后才统一写报告。如果中途超时，就没有 partial report。

这是一个工程改进点：

```text
下一步应给 eval 增加 incremental save / resume
```

也就是说，未来完整 200 条 eval 应该边跑边保存：

```text
每完成一条 trajectory -> append jsonl
定期刷新 report
支持从已有 trajectory 继续跑
```

这样长评估即使中断，也不会丢结果。

## 9. 面试时怎么讲

可以这样讲：

> 在第一版 SFT 证明 tool-call 格式能学会之后，我把训练扩展到 800 条 expert trajectories，共 2530 个 next-action 样本，训练 200 个 optimizer steps。这个 LoRA 只训练约 440 万参数，占总参数 0.88%，在本机 12GB 显存上约 5 分钟完成。结果上，Base model 的第一步 tool-call format accuracy 是 0，而 SFT 在 200 条 eval 上达到 1.0。完整 rollout 上，上一版 step40 的成功率是 15%，step200 在 50 条 eval 上达到 64%，并且 invalid tool call rate 只有 1.85%。这说明 SFT 不只是学会了 wrapper，也初步学会了多步网页操作流程。

如果被问为什么还没跑完整 200 条 full eval：

> 我尝试过 200 条 full rollout，但当前 eval 是结束后统一写报告，长评估超时会丢结果。所以我先用 50 条 full rollout 得到可靠阶段性指标，下一步会给 eval 加 incremental save/resume，再跑完整 200 条。

## 10. 当前局限

当前结果仍然有局限：

- 50 条 full rollout 还不是完整 200 条 eval。
- 任务仍然来自模板生成，泛化难度有限。
- 有 36% 任务失败，需要 error analysis。
- invalid tool call 虽低，但不是 0。
- 还没有 reward function 和 GRPO。

## 11. 下一步建议

下一步最应该做的是：

```text
Error analysis + incremental eval
```

具体包括：

1. 写 error classifier：
   - 过早 submit
   - 选错 element_id
   - 选错答案
   - 重复点击
   - invalid tool call
2. 分析 50 条 full rollout 的失败案例。
3. 给 eval 加增量保存和 resume。
4. 重新跑完整 200 条 full eval。
5. 如果 SFT 的失败集中在答案选择，再考虑 reward/path 设计和 GRPO。

这一步之后，项目已经可以比较有底气地说：

```text
Base -> SFT 的工具调用能力有明显提升
SFT 已经能在本地 WebNav 环境中完成相当比例的任务
```
