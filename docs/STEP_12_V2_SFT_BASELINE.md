# Step 12: V2 SFT Held-Out Layout Baseline

本步骤训练第一个 V2 LoRA SFT，并在完全未见的 layout C、全新随机 element ID 上评测 500 条任务。

## 1. 训练设置

```text
base model: Qwen2.5-0.5B-Instruct
train trajectories: 3000
next-action examples: 11200
optimizer steps: 1400
batch size: 2
gradient accumulation: 4
effective batch size: 8
max sequence length: 2048
learning rate: 2e-4
LoRA rank: 8
```

Loss 从 0.996 降到 0.0086，训练过程中没有 NaN、Inf 或显存错误。

Checkpoint：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_v2_step1400
```

## 2. Held-Out Eval

评测条件：

- 500 tasks。
- layout C 从未出现在训练中。
- train/eval element ID overlap 为 0。
- temperature 0。
- max new tokens 96。

总体结果：

| Metric | Result |
| --- | ---: |
| Task success | 156/500 |
| Task success rate | 31.2% |
| Final answer accuracy | 31.2% |
| Tool-call format accuracy | 99.95% |
| Invalid tool-call rate | 4.79% |
| Average model steps | 3.84 |
| Submitted | 487 |
| Max steps | 13 |

## 3. Difficulty Breakdown

| Difficulty | Success | Rate |
| --- | ---: | ---: |
| Easy | 103/132 | 78.0% |
| Medium | 13/66 | 19.7% |
| Hard | 40/302 | 13.2% |

模型已经学会协议和简单实体 grounding，但复杂度增加后成功率快速下降。

## 4. Error Breakdown

```text
wrong_candidate_after_filter: 300
wrong_click_path: 30
invalid_tool_call: 13
format_error: 1
success: 156
```

`wrong_candidate_after_filter` 占全部失败的 87.2%，是压倒性的主要瓶颈。

## 5. Template Results

| Template | Success rate |
| --- | ---: |
| Shopping name | 97.0% |
| Course title | 93.9% |
| Course code | 69.7% |
| Shopping price lookup | 51.5% |
| Department + credits + highest | 33.3% |
| Category lowest price | 32.4% |
| Color + category | 21.2% |
| Category + budget + highest | 20.6% |
| Department + time | 18.2% |
| Credits + department | 17.6% |
| Department highest rating | 15.2% |
| Credits highest rating | 0.0% |
| Category highest rating | 0.0% |
| Under $100 highest rating | 0.0% |
| Under $100 lowest price | 0.0% |

## 6. Behavior Funnel

新增 `eval/v2_behavior_analysis.py`，从真实 action trajectory 分析筛选与候选点击：

```text
filtered tasks: 368
correct filters: 353
correct filter rate: 95.9%
candidate attempts after correct filter: 353
correct candidates after correct filter: 53
candidate accuracy after correct filter: 15.0%
```

这个 funnel 很重要：模型几乎总能理解应该点击哪个筛选控件，但在看到筛选结果后无法可靠选择正确候选。

## 7. Position Shortcut

模型候选位置分布：

```text
position 1: 135
position 2: 108
position 3: 5
position 4: 70
position 9: 34
position 12: 1
```

300 个 wrong-candidate failures 中：

- 111 次选择 position 1，占 37%。
- 100 次选择 position 2。
- 55 次选择 position 4。
- 31 次选择 position 9。

这不是单一的“总点第一个”，而是模板相关的位置 shortcut：

- under-$100 highest 与 lowest 两个模板始终选择 position 2。
- shopping category 模板主要选择 position 1 或 4。
- course 模板主要选择 position 1、2、4 或 9。

训练布局中的候选位置与答案仍然相关，模型学到了位置模式，而不是对 rating/price/department 属性做稳定比较。

## 8. 研究结论

V2 baseline 将能力拆得更清楚：

1. 工具格式已解决：99.95%。
2. 未见 ID 的直接实体 grounding 基本可行：name/title 超过 93%。
3. 筛选控件选择基本可行：95.9%。
4. 筛选后的候选比较失败：15.0%。
5. 失败主要来自位置 shortcut，而不是 JSON 格式。

因此 31.2% 不能简单解释为“模型不会 V2”。模型已经学会了协议、过滤和实体定位，但尚未学会结构无关的候选比较。

## 9. 为什么暂不进入 V2 GRPO

四个核心 ranking 模板成功率为 0，说明基础 policy 在这些任务上没有可靠行为。直接 GRPO 可能强化随机位置选择，并再次产生较大的 seed 方差。

更合理的顺序是：

```text
V2.1 candidate-shuffled SFT
-> held-out structural eval
-> 确认 ranking baseline 非零且稳定
-> 再做 GRPO
```

## 10. V2.1 设计方向

1. 为同一候选集合生成大量随机顺序和随机 ID 页面实例。
2. 让同一答案在训练中出现在不同 position，解除 position-answer 相关性。
3. 增加 counterfactual pairs：属性不变但位置变化。
4. 保持 eval 使用独立随机种子和未见布局。
5. targeted oversampling highest/lowest 与多条件模板。
6. 在训练后先检查 candidate-position distribution，再决定是否进入 RL。

## 11. 产物

```text
outputs/eval_reports/v2_sft_step1400_eval500_report.json
outputs/eval_reports/v2_sft_step1400_error_analysis.json
outputs/eval_reports/v2_sft_step1400_training_metadata.json
outputs/eval_reports/v2_sft_step1400_behavior_analysis.json
outputs/eval_reports/v2_sft_step1400_behavior_analysis.md
artifacts/server_runs/2026-06-24-v2/
```

## 12. 面试表达

> V2 SFT 在未见布局上的总成功率是 31.2%，但工具格式保持 99.95%。进一步做 action funnel 后发现，筛选控件准确率达到 95.9%，真正的问题是筛选后候选准确率只有 15%。位置分布显示模型在不同模板中重复选择固定位置，证明它学到了 position shortcut。基于这个结果，我没有直接继续做 RL，而是把下一轮设计改成 candidate-shuffled counterfactual SFT，让答案与位置去相关。
