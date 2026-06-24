# WebNav-RL 最终项目报告

## 1. 项目目标

WebNav-RL 的目标是用一个小而完整的系统研究网页导航智能体的 post-training：模型不仅回答问题，还要读取环境反馈、调用工具、点击页面元素并提交答案。

项目刻意使用本地生成页面和可验证任务，原因是：

- 页面状态稳定，可重复运行。
- 每个任务都有标准答案和 expert click path。
- 可以严格区分格式错误、非法动作、点击错误和最终答案错误。
- 能在消费级 GPU 上完成从 SFT 到 RL 的完整实验。

## 2. 系统架构

完整数据流：

```text
page generator
-> task generator
-> deterministic browser environment
-> expert trajectories
-> next-action SFT data
-> LoRA SFT
-> full model/tool rollout evaluation
-> error analysis
-> shaped reward
-> grouped stochastic rollouts
-> group-relative advantages
-> GRPO-KL LoRA update
-> paired multi-seed evaluation
```

核心模块：

| Module | Responsibility |
| --- | --- |
| `pages/page_generator.py` | 生成购物和课程页面及 metadata |
| `env/browser_env.py` | 执行 open/click/read/submit 工具动作 |
| `env/verifier.py` | exact-match 最终答案验证 |
| `tasks/task_generator.py` | 从 metadata 生成 train/eval 任务 |
| `rollout/model_runner.py` | 模型生成、工具执行和状态更新循环 |
| `training/sft_train.py` | 专家 next-action LoRA SFT |
| `eval/error_analysis.py` | 行为级失败分类 |
| `training/reward.py` | 格式、路径、答案和惩罚组合 reward |
| `training/grpo_rollout.py` | grouped rollout 和 relative advantage |
| `training/grpo_train.py` | advantage-weighted policy loss + reference KL |
| `eval/multiseed_analysis.py` | 多 seed 配对、McNemar 和模板分析 |

## 3. 数据和切分

任务总数 1000：

- 800 train tasks。
- 200 held-out eval tasks。

SFT 使用 train expert trajectories。正式 GRPO rollout 也只从 `tasks/train_tasks.jsonl` 采样；所有最终指标来自独立的 `tasks/eval_tasks.jsonl`。

实验中曾发现早期服务器脚本误用 eval split 采样 rollout，随后立即纠正，并重新从 train split 采样。误采样数据只用于流程验证，没有进入最终模型或最终结论。

## 4. SFT 阶段

Base model 是 Qwen2.5-0.5B-Instruct。

Base model 在 20-task first-step 检查中：

- tool-call format accuracy 0%。
- invalid tool-call rate 100%。

说明通用 chat model 没有自然对齐到项目要求的 XML/JSON action protocol。

SFT 配置：

- 800 条 expert trajectories。
- 2530 个 next-action examples。
- LoRA rank 8，约 4.4M 可训练参数。
- 200 optimizer steps。

SFT 后 first-step format accuracy 达到 100%，说明 SFT 首先解决了“模型是否会按协议行动”的问题。

## 5. Reward 设计

Reward 由六部分组成：

```text
+0.2  all outputs parse as valid tool calls
+0.3  expert click-path prefix score
+0.4  exact final answer
-0.05 extra step beyond expert budget
-0.2  each invalid tool call
-0.2  episode without submit
```

在 SFT 轨迹上，成功样本平均 reward 为 0.876，失败样本为 0.221，说明 reward 能区分轨迹质量，而不是只产生稀疏的 0/1 信号。

## 6. 正式 GRPO-KL 实验

服务器：RTX 5090 32GB。

Grouped rollout：

- 100 train tasks。
- group size 4。
- 400 trajectories。
- temperature 0.7。
- mean reward 0.5925。
- 40 个 nonzero-advantage groups。
- 506 个 nonzero-advantage assistant actions。

Policy update：

```text
loss = advantage_weighted_policy_loss + beta * reference_kl
beta = 0.02
learning_rate = 1e-5
```

Frozen reference 使用训练前的 SFT adapter，用于限制 RL policy drift。

## 7. 最终结果

同服务器、同 SFT checkpoint、同 200-task eval split：

| Run | Successes | Success rate | Delta | Invalid rate | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 121 | 60.5% | - | 1.57% | - |
| GRPO seed 7 | 128 | 64.0% | +3.5 pp | 1.57% | 0.1435 |
| GRPO seed 17 | 126 | 63.0% | +2.5 pp | 1.88% | 0.4049 |
| GRPO seed 29 | 119 | 59.5% | -1.0 pp | 1.74% | 0.8318 |

跨 seed 汇总：

- mean success rate：62.17%。
- sample standard deviation：2.36 pp。
- mean delta vs SFT：+1.67 pp。
- 3 个 seed 中 2 个提升、1 个退化。
- 所有 run 的 tool-call format accuracy 都是 100%。

结论：当前结果是初步正向信号，但不是稳定显著提升。

## 8. 配对与稳定性分析

seed 7：

- 12 个任务从错误变正确。
- 5 个任务从正确变错误。
- 净提升 7。
- McNemar exact p=0.1435。

跨三个 seed：

- 11 个任务在多数 seed 下稳定改善。
- 6 个任务在多数 seed 下稳定退化。

这比只看平均成功率更重要，因为它说明模型不是简单地随机交换相同数量的正确任务，而是在部分任务上形成了可重复的行为变化。

## 9. 模板级发现

跨 seed 平均变化：

| Template | Delta |
| --- | ---: |
| `course_title` | +21.67 pp |
| `course_4_credit_department` | +11.11 pp |
| `shopping_name` | +9.88 pp |
| `shopping_category_lowest_price` | +6.06 pp |
| `course_code` | -6.41 pp |
| `course_department_highest_rating` | -36.36 pp |

解释：

- 当前 reward 对直接实体定位和基础点击路径帮助较明显。
- 排序/筛选后的候选比较没有整体改善。
- department + highest-rating 组合任务明显退化，说明路径模仿 reward 不等于稳定的多条件推理能力。

## 10. Full-Pass 对照

使用 batch size 4、127 steps 覆盖全部 506 个 action examples 后：

- success rate 仍为 64.0%。
- invalid tool calls 从 SFT 的 10 增加到 13。

因此 full-pass 没有超过 100-step checkpoint，反而轻微损害动作合法性。它说明在固定离线 rollout 上增加 exposure 可能带来 policy drift，而不是持续收益。

## 11. 工程与研究判断

项目中几项重要判断：

1. 先搭确定性环境，再训练模型，使失败可以定位。
2. SFT 和 RL 使用严格 train/eval split，避免数据泄漏。
3. 评测支持 incremental/resume，长任务中断不会丢失进度。
4. 不只报告 best seed，同时报告 mean/std 和退化 seed。
5. 不把不显著的 +3.5 pp 包装成确定结论。
6. 通过 full-pass 对照发现训练更多并不必然更好。

## 12. 局限性

- 页面类型和布局仍少，可能学习固定 element-id 模式。
- 任务模板数量有限，真实网页泛化尚未验证。
- 60% rollout groups 没有 advantage，数据效率偏低。
- GRPO trainer 是简化实现，没有完整 PPO ratio clipping。
- 当前只有 200 个 eval tasks，统计功效有限。
- 多条件候选比较仍是明显瓶颈。

## 13. 下一步

优先级从高到低：

1. 增加页面布局、实体和干扰项，建立结构级 held-out split。
2. 增加针对排序、筛选和多条件比较的数据与过程 reward。
3. 提升 group 内多样性，减少 zero-advantage groups。
4. 加入 early stopping、KL drift 聚合日志和 checkpoint selection。
5. 数据和 reward 改善后，再迁移到 Qwen 1.5B 做规模验证。

## 14. 最终一句话

WebNav-RL 已完成从可验证环境、专家数据、LoRA SFT 到 GRPO-KL 和多 seed 统计分析的端到端闭环；最佳 run 从 60.5% 提升到 64.0%，三 seed 平均为 62.17%，证明存在正向信号，同时也清楚暴露了训练方差、zero-advantage 数据和多条件候选选择三个下一阶段问题。
