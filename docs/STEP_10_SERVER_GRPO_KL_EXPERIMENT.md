# Step 10: Server GRPO-KL Experiment

本步骤在 RTX 5090 32GB 服务器上完成了正式的 GRPO-KL 数据采样、训练和独立评测。

## 1. 实验设置

Base model：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

SFT：

- 800 条训练轨迹。
- 2530 个 next-action examples。
- LoRA rank 8。
- 200 optimizer steps。
- learning rate `2e-4`。

GRPO rollout：

- rollout split：`tasks/train_tasks.jsonl`。
- 100 个训练任务。
- group size 4。
- 共 400 条 rollout。
- temperature `0.7`。
- 40 个 group 有非零 advantage。
- 506 个 nonzero-advantage action examples。

独立评测：

- eval split：`tasks/eval_tasks.jsonl`。
- 200 个任务。
- temperature `0.0`。
- SFT 和 GRPO-KL 使用完全相同的评测配置。

## 2. Rollout 数据质量

训练集 grouped rollout 汇总：

```json
{
  "num_samples": 400,
  "num_groups": 100,
  "mean_reward": 0.5925,
  "success_rate": 0.5625,
  "mean_abs_advantage": 0.106875,
  "nonzero_advantage_groups": 40,
  "zero_advantage_groups": 60
}
```

40% 的 group 产生了组内 reward 差异，说明采样数据包含可用于 relative policy update 的偏好信号。60% 的 group 内所有样本 reward 相同，不贡献 GRPO advantage gradient。

## 3. 100-Step GRPO-KL

训练配置：

- batch size 1。
- 100 optimizer steps。
- learning rate `1e-5`。
- KL beta `0.02`。
- frozen reference 为 SFT step200 adapter。

评测结果：

| Metric | SFT | GRPO-KL step100 | Delta |
| --- | ---: | ---: | ---: |
| Successes | 121 | 128 | +7 |
| Task success rate | 60.5% | 64.0% | +3.5 pp |
| Tool-call format accuracy | 100% | 100% | 0 |
| Invalid tool-call rate | 1.57% | 1.57% | 0 |
| Average model steps | 3.185 | 3.180 | -0.005 |

这是本轮实验的最佳 checkpoint：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_grpo_kl_task100_group4_step100
```

## 4. 配对分析

同一批 200 个任务上的配对结果：

```text
wrong -> correct: 12
correct -> wrong: 5
both correct: 116
both wrong: 67
McNemar exact p-value: 0.14346
```

因此：

- 净提升 7 题。
- 错变对数量是对变错的 2.4 倍。
- 方向为正，但 `p > 0.05`，当前单次实验没有达到统计显著性。

严谨表述：

> GRPO-KL 在一次独立评测中将任务成功率从 60.5% 提升到 64.0%，且没有破坏工具格式或增加总体非法调用率。配对分析显示 12 个任务得到修复、5 个任务发生退化，但 McNemar 检验尚未达到统计显著性，因此该结果应视为初步正向证据，而不是已经确认的稳定提升。

## 5. 错误类型变化

| Error type | SFT | GRPO-KL step100 | Delta |
| --- | ---: | ---: | ---: |
| Wrong click path | 61 | 52 | -9 |
| Wrong candidate after filter | 16 | 16 | 0 |
| Invalid tool-call failure | 2 | 4 | +2 |
| Success | 121 | 128 | +7 |

主要提升来自 `wrong_click_path` 减少，说明当前 expert-path reward 对基础点击路径选择有效。

`wrong_candidate_after_filter` 没有变化，说明筛选、排序、多候选比较仍是主要能力瓶颈。后续 reward 或训练数据应该更直接覆盖候选比较，而不是只继续加强路径模仿。

## 6. Full-Pass 对照实验

为了覆盖全部 506 个有效 action examples，又训练了一个 full-pass adapter：

- batch size 4。
- 127 optimizer steps。
- 基本覆盖全部 506 个样本。

结果：

| Metric | SFT | GRPO-KL fullpass | Delta |
| --- | ---: | ---: | ---: |
| Successes | 121 | 128 | +7 |
| Task success rate | 60.5% | 64.0% | +3.5 pp |
| Tool-call format accuracy | 100% | 100% | 0 |
| Invalid tool-call rate | 1.57% | 2.05% | +0.48 pp |
| Invalid tool calls | 10 | 13 | +3 |
| Average model steps | 3.185 | 3.170 | -0.015 |

Full-pass 没有比 step100 多完成任务，却增加了非法工具调用，因此不作为最佳模型。

这说明在当前小规模、离线 rollout 数据上，更多 optimizer exposure 不一定更好。训练继续推进后，policy drift 增加，但 reward signal 没有提供额外任务收益。

## 7. KL 数值观察

训练中 KL 大部分保持较小，但 full-pass 后段在个别 batch 出现约 `0.18-0.24` 的局部峰值。乘以 `kl_beta=0.02` 后仍未导致 loss 爆炸，但与非法调用轻微增加的现象一致，说明后段 policy drift 更明显。

日志还出现了约 `-1e-5` 的 KL 数值。k3 KL estimator 理论上非负，这是 FP16 消减误差。后续代码已调整为用 FP32 计算 k3 estimator，并将最终值 clamp 到非负。

## 8. 项目结论

本步骤证明：

1. Train/eval split 已严格分离，避免 RL 数据泄漏。
2. Group rollout、reward、advantage、reference KL、LoRA update 和独立评测完成闭环。
3. GRPO-KL step100 获得 +3.5 个百分点的初步提升。
4. 提升主要来自基础点击路径错误减少。
5. 继续 full-pass 训练没有额外收益，并轻微损害动作合法性。

当前推荐保留：

```text
SFT baseline:
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200

Best GRPO-KL:
outputs/checkpoints/qwen2_5_0_5b_lora_grpo_kl_task100_group4_step100
```

## 9. Multi-Seed 复现

为了检验单次 +3.5 pp 是否稳定，又使用相同 SFT adapter、相同 rollout 数据和相同超参数，补做了 seed 17 和 seed 29。

| Training seed | Successes | Success rate | Delta vs SFT | Invalid tool-call rate |
| --- | ---: | ---: | ---: | ---: |
| SFT baseline | 121 | 60.5% | - | 1.57% |
| GRPO-KL seed 7 | 128 | 64.0% | +3.5 pp | 1.57% |
| GRPO-KL seed 17 | 126 | 63.0% | +2.5 pp | 1.88% |
| GRPO-KL seed 29 | 119 | 59.5% | -1.0 pp | 1.74% |

三个 GRPO-KL seed 的结果：

```text
mean success rate: 62.17%
sample standard deviation: 2.36 percentage points
mean delta vs SFT: +1.67 percentage points
```

自动生成的完整配对与模板分析：

```text
outputs/eval_reports/grpo_multiseed_analysis.json
outputs/eval_reports/grpo_multiseed_analysis.md
```

跨 seed 多数投票显示：

- 11 个任务在至少 2/3 seed 中稳定从错变对。
- 6 个任务在至少 2/3 seed 中稳定从对变错。
- `course_title` 平均提升 21.67 pp。
- `shopping_name` 平均提升 9.88 pp。
- `course_department_highest_rating` 平均下降 36.36 pp。

解释：

- 3 个 seed 中有 2 个优于 SFT，1 个低于 SFT。
- 平均值方向仍为正，但收益较小且 seed 方差明显。
- seed 7 的 64.0% 是 best observed checkpoint，不代表稳定期望性能。
- 当前结果支持“GRPO-KL 有初步正向信号”，不支持“GRPO-KL 已稳定显著提升”。

面试时应这样说：

> 最佳单次 GRPO-KL 将成功率从 60.5% 提高到 64.0%。为了避免只报告最好结果，我又补了两个训练 seed，三次平均为 62.17%，其中两个 seed 提升、一个 seed 退化。这说明方法存在正向信号，但训练方差仍然较大，下一步重点应该是提高 rollout 多样性和有效 advantage group 比例，而不是继续挑选最好 seed。

## 10. 下一步

下一轮最有价值的工作不是盲目增加训练步数，而是：

- 为排序/筛选后的候选比较增加专门任务和 reward。
- 扩大有效 rollout 覆盖，降低训练 seed 方差。
- 保存 best checkpoint，并引入按 eval 或 KL 的 early stopping。
- 增加 group 内多样性，减少 60% zero-advantage groups。
