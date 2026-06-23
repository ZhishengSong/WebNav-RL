# Step 08: Minimal GRPO Prototype

本步骤实现了一个最小 GRPO-style LoRA update。

重要说明：

> 这一步的目标是打通“group rollout -> reward -> advantage -> logprob loss -> LoRA update -> eval”的训练闭环，不是证明 GRPO 已经提升最终指标。

当前结果是：GRPO prototype 能正常训练和评估，但因为只用了 4 个 task 的 16 条 rollout、更新 5 步，20 条 eval 成功率从 SFT 的 70% 降到 45%。这不是失败，而是一个很正常的 prototype 结果：闭环通了，但数据太小、没有 KL/reference、容易扰动策略。

## 1. 新增文件

```text
training/grpo_train.py
```

作用：

- 读取 `training/grpo_rollout.py` 生成的 group rollout records。
- 把每条 trajectory 拆成 assistant action 训练样本。
- 使用每条 trajectory 的 group-relative advantage。
- 计算 sampled tool-call 的 token logprob。
- 用 advantage 加权 policy gradient loss。
- 更新 LoRA adapter。
- 保存新的 GRPO prototype adapter。

## 2. 输入数据

使用上一阶段生成的 group rollout：

```text
outputs/rollouts/grpo_sft_step200_group4_task4.jsonl
```

数据规模：

```text
tasks: 4
group_size: 4
samples: 16 trajectories
nonzero_advantage_examples: 40 assistant actions
```

Group rollout 报告：

```text
outputs/eval_reports/grpo_sft_step200_group4_task4_report.json
```

关键统计：

```text
num_samples: 16
num_groups: 4
success_rate: 0.5625
mean_reward: 0.565625
mean_abs_advantage: 0.17656
nonzero_advantage_groups: 3
```

这说明 rollout 中确实有好坏差异，能产生 GRPO 所需的正负 advantage。

## 3. Loss 设计

当前最小 loss：

```text
loss = advantage * cross_entropy(sampled_action)
```

它等价于：

```text
policy_gradient_loss = - advantage * logprob(sampled_action)
```

直观解释：

- advantage > 0：降低 CE，也就是提高这条 action 的概率。
- advantage < 0：反向推动，降低这条 action 的概率。
- advantage = 0：不参与训练。

当前实现是 prototype，没有加入：

- PPO/GRPO clipping。
- reference model KL。
- token-level KL。
- length normalization 之外的复杂项。

因此它是最小可运行训练闭环，而不是完整论文级 GRPO。

## 4. 训练命令

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe training\grpo_train.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step200 `
  --rollouts outputs\rollouts\grpo_sft_step200_group4_task4.jsonl `
  --output-dir outputs\checkpoints\qwen2_5_0_5b_lora_grpo_proto_step5 `
  --max-steps 5 `
  --max-seq-len 1024 `
  --batch-size 1 `
  --gradient-accumulation-steps 1 `
  --learning-rate 5e-5 `
  --log-every 1
```

输出 adapter：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_grpo_proto_step5
```

训练 metadata：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_grpo_proto_step5/grpo_training_metadata.json
```

## 5. 训练日志

```json
[
  {
    "optimizer_step": 1,
    "loss": -0.0224,
    "mean_advantage": -0.35
  },
  {
    "optimizer_step": 2,
    "loss": -0.00038,
    "mean_advantage": -0.4125
  },
  {
    "optimizer_step": 3,
    "loss": -0.00014,
    "mean_advantage": -0.15
  },
  {
    "optimizer_step": 4,
    "loss": 0.00478,
    "mean_advantage": 0.1375
  },
  {
    "optimizer_step": 5,
    "loss": 0.00016,
    "mean_advantage": 0.1375
  }
]
```

Loss 有正有负是正常的，因为 advantage 可以为负。负 advantage 样本会推动模型降低对应 action 的概率。

## 6. Eval 结果

评估命令：

```powershell
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_grpo_proto_step5 `
  --tasks tasks\eval_tasks.jsonl `
  --limit 20 `
  --output outputs\trajectories\grpo_proto_step5_eval20_full_trajectories.jsonl `
  --report outputs\eval_reports\grpo_proto_step5_eval20_full_report.json `
  --failures outputs\eval_reports\grpo_proto_step5_eval20_full_failures.jsonl `
  --device auto `
  --max-new-tokens 48 `
  --temperature 0.0
```

结果：

```json
{
  "num_tasks": 20,
  "successes": 9,
  "task_success_rate": 0.45,
  "tool_call_format_accuracy": 1.0,
  "invalid_tool_call_rate": 0.0303,
  "average_model_steps": 3.3
}
```

对比 SFT step200 同样 20 条：

```json
{
  "num_tasks": 20,
  "successes": 14,
  "task_success_rate": 0.70,
  "tool_call_format_accuracy": 1.0,
  "invalid_tool_call_rate": 0.0303,
  "average_model_steps": 3.3
}
```

## 7. 如何解释指标下降

这一步成功率下降是可以接受的，因为：

- 只用了 4 个 task 的 rollout。
- 只有 16 条 trajectories。
- 只做了 5 个 optimizer steps。
- 没有 KL/reference 约束。
- 没有 PPO/GRPO clipping。
- 数据分布太窄，容易过拟合少数 sampled actions。

这一步的价值是：

```text
训练闭环打通了
LoRA adapter 能被 GRPO-style loss 更新
更新后的 adapter 能正常保存和 eval
tool-call format accuracy 没有崩掉
```

也就是说，它是工程可行性验证，不是最终 RL 实验结论。

## 8. 面试时怎么讲

可以这样讲：

> 我实现了一个最小 GRPO prototype。它读取同一任务多条 sampled trajectories，用 rule-based reward 计算每条轨迹的 reward，再减去 group mean 得到 advantage。训练时把每条 trajectory 拆成 assistant action，计算 sampled tool-call 的 logprob，用 `-advantage * logprob` 做 policy gradient 更新 LoRA。小规模实验能完成参数更新和保存，并能重新 eval。因为 prototype 只用了 4 个 task、16 条 rollout，没有 KL 和 clipping，所以指标没有提升，20 条 eval 成功率从 SFT 的 70% 降到 45%。这说明下一步需要扩大 rollout 数据并加入 KL/reference 约束，而不是说明 GRPO 思路无效。

如果被问“为什么不直接大规模跑 GRPO”：

> 我先把最小闭环拆出来验证。GRPO 涉及 rollout、reward、advantage、logprob、KL、LoRA 更新多个环节。如果一上来大规模训练，指标出问题很难定位。现在最小闭环能跑通，下一步再逐步加 KL、clipping 和更大的 group rollout。

## 9. 当前局限

当前 GRPO prototype 的限制：

- rollout 数据太小。
- 没有 reference model KL。
- 没有 clipping。
- 没有按 token/response 长度做更细的标准化。
- 没有批量 rollout 优化。
- 没有完整 200 条 eval 对比。

## 10. 下一步建议

下一步可以做两个方向：

### 方向 A：稳定 GRPO 算法

加入：

```text
reference logprob
KL penalty
advantage normalization
loss clipping
larger group rollout
```

### 方向 B：更务实的项目推进

先不追 RL 指标，补项目文档和实验表格：

```text
Base vs SFT vs GRPO-proto 对比
error analysis 图表
reward breakdown 表
README 完整化
简历描述
```

从简历项目角度看，当前已经有：

```text
环境
SFT
eval
error analysis
reward
GRPO prototype
```

已经可以整理成一个相当完整的 Agentic RL post-training 项目。
