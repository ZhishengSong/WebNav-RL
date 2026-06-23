# Step 07: GRPO Group Rollout Data

本步骤实现了 GRPO 前置的数据结构：

```text
同一个 task
-> 采样多条 trajectory
-> 每条 trajectory 算 reward
-> 组内计算 relative advantage
```

这一步还没有做参数更新，也就是还没有真正训练 GRPO；它完成的是 GRPO 训练前最关键的 rollout/reward/advantage 数据链路。

## 1. 为什么先做 group rollout

GRPO 的核心不是只看单条 response 的 reward，而是对同一个 prompt/task 采样一组 responses，然后用组内相对表现来计算 advantage。

直观地说：

```text
同一个任务采样 4 条
其中 2 条答对，2 条答错
答对的 reward 高于组均值 -> positive advantage
答错的 reward 低于组均值 -> negative advantage
```

这样训练时可以鼓励模型增加高 advantage 轨迹的概率，降低低 advantage 轨迹的概率。

## 2. 新增文件

```text
training/grpo_rollout.py
```

功能：

- 读取 task 文件。
- 对每个 task 采样 `group_size` 条 trajectory。
- 复用现有 `run_model_task(...)` 执行环境交互。
- 复用 `training/reward.py` 计算 reward breakdown。
- 计算每条样本的：
  ```text
  advantage = reward - group_mean_reward
  ```
- 保存 group rollout JSONL。
- 输出 group reward/advantage report。

## 3. 输出数据格式

每行是一个 sample，而不是一个 task。

示例字段：

```json
{
  "group_id": "shop_00801::g00001",
  "task_id": "shop_00801",
  "sample_index": 0,
  "group_size": 4,
  "group_mean_reward": 0.2,
  "advantage": 0.0,
  "reward": {
    "total_reward": 0.2,
    "error_type": "wrong_click_path"
  },
  "trajectory": {
    "messages": [],
    "actions": [],
    "summary": {}
  }
}
```

这个结构后续可以直接作为 GRPO trainer 的 rollout buffer。

## 4. Expert replay smoke test

先用 expert replay 检查 group/reward/advantage 结构。

命令：

```powershell
python training\grpo_rollout.py `
  --expert-replay `
  --tasks tasks\eval_tasks.jsonl `
  --limit 2 `
  --group-size 3 `
  --output outputs\rollouts\grpo_expert_smoke.jsonl `
  --report outputs\eval_reports\grpo_expert_smoke_report.json
```

结果：

```json
{
  "num_samples": 6,
  "num_groups": 2,
  "group_size": 3,
  "mean_reward": 0.9,
  "success_rate": 1.0,
  "mean_abs_advantage": 0.0,
  "nonzero_advantage_groups": 0,
  "zero_advantage_groups": 2
}
```

解释：

- expert replay 每条都是同样正确路径。
- 同组 reward 全相同。
- advantage 全是 0。
- 这说明 reward 和 group mean 计算正确。

## 5. 真实 SFT 模型 group rollout

使用 SFT step200 adapter：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200
```

命令：

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe training\grpo_rollout.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step200 `
  --tasks tasks\eval_tasks.jsonl `
  --limit 4 `
  --group-size 4 `
  --temperature 0.7 `
  --max-new-tokens 48 `
  --output outputs\rollouts\grpo_sft_step200_group4_task4.jsonl `
  --report outputs\eval_reports\grpo_sft_step200_group4_task4_report.json `
  --incremental `
  --report-every 1
```

结果：

```json
{
  "num_samples": 16,
  "num_groups": 4,
  "group_size": 4,
  "mean_reward": 0.565625,
  "min_reward": 0.2,
  "max_reward": 0.9,
  "success_rate": 0.5625,
  "mean_abs_advantage": 0.1765625,
  "nonzero_advantage_groups": 3,
  "zero_advantage_groups": 1
}
```

输出：

```text
outputs/rollouts/grpo_sft_step200_group4_task4.jsonl
outputs/eval_reports/grpo_sft_step200_group4_task4_report.json
```

## 6. Advantage 示例

采样得到的 rewards：

```text
[0.2, 0.2, 0.2, 0.2,
 0.2, 0.9, 0.9, 0.2,
 0.9, 0.9, 0.6, 0.6,
 0.9, 0.9, 0.35, 0.9]
```

对应 advantages：

```text
[0.0, 0.0, 0.0, 0.0,
 -0.35, 0.35, 0.35, -0.35,
 0.15, 0.15, -0.15, -0.15,
 0.1375, 0.1375, -0.4125, 0.1375]
```

这说明：

- 第一个 group 全部一样差，所以 advantage 全是 0。
- 其他 group 内有好坏差异，能产生正负 advantage。
- GRPO 后续可以用这些 advantage 更新模型。

## 7. 为什么这个结果重要

此前我们已有：

```text
SFT model
reward function
full eval report
error analysis
```

但还缺 GRPO 最核心的数据形态：

```text
same prompt -> multiple sampled trajectories -> relative rewards
```

现在这条链路已经打通了。

这意味着下一步可以开始做真正的 GRPO prototype：

```text
采样 group trajectories
计算 logprobs
计算 group-relative advantage
反向传播更新 LoRA
```

## 8. 面试时怎么讲

可以这样讲：

> 在 SFT 模型和 reward function 稳定后，我实现了 GRPO 前置的 group rollout pipeline。对同一个 task 采样多条 trajectory，每条 trajectory 用 rule-based reward function 打分，然后计算 group mean reward 和 advantage。小规模实验中，4 个 task、每个 task 采样 4 条，共 16 条 trajectory，其中 3 个 group 产生了非零 advantage。这证明模型在 temperature sampling 下会产生质量不同的轨迹，reward 能区分它们，后续可以把这些 advantage 接入 GRPO 训练。

如果被问为什么还没有直接训练 GRPO：

> 我先把 rollout buffer 和 reward/advantage 计算单独验证。GRPO 的训练更新涉及 logprob、KL、LoRA 参数更新，调试成本更高；如果 rollout/reward 数据本身不可靠，直接训练很难定位问题。现在数据链路打通后，再接 trainer 更稳。

## 9. 当前局限

当前 group rollout 仍是 prototype：

- 只跑了 4 个 task。
- group_size 是 4。
- 只计算 reward 和 advantage，还没有 logprob。
- 没有 reference model KL。
- 没有真正 optimizer update。
- group rollout 速度还比较慢。

这些都是下一步 GRPO trainer 要解决的问题。

## 10. 下一步

下一步建议：

```text
实现最小 GRPO trainer prototype
```

最小版本可以先做：

1. 对每条 sampled assistant output 计算 logprob。
2. 使用当前 LoRA model 作为 policy。
3. 可选：先不加 KL，或使用 SFT adapter 作为 reference。
4. 使用 advantage 加权 policy gradient loss。
5. 小规模更新 10-20 steps。
6. 重新 eval，看 success rate 或 invalid tool call 是否变化。

如果想更稳，也可以先做：

```text
rollout speed optimization + larger group rollout report
```

比如：

```text
limit 20 tasks
group_size 4
temperature 0.7
```

但从项目主线看，现在已经可以进入 GRPO prototype。
