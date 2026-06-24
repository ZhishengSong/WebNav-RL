# WebNav-RL 面试问答与简历表述

## 30 秒版本

> 我做了一个小模型网页导航 post-training 项目。先用程序生成可验证的购物和课程页面、任务、专家路径，再训练 Qwen2.5-0.5B 的 LoRA SFT，让模型通过工具调用完成多步导航。之后我实现了错误分析、shaped reward、group rollout 和带 reference KL 的 GRPO 更新。受控 200 题评测中，SFT 是 60.5%，最佳 GRPO 是 64.0%；三 seed 平均 62.17%，所以我把结果表述为初步正向但仍有训练方差。

## 2 分钟版本

> 这个项目关注的是 agent post-training，而不是普通文本分类。模型每一步只能输出一个严格工具调用，环境执行 open、click、read 或 submit，再把 observation 返回给模型。
>
> 我先搭了一个确定性本地环境，生成 1000 个任务并划分 800 train、200 eval。规则专家生成完整轨迹，我把每个 assistant action 转成 next-action SFT 样本，对 Qwen2.5-0.5B 做 LoRA。Base model 的工具格式准确率是 0%，SFT 后达到 100%，说明 action protocol 对齐成功。
>
> 完整 rollout 后我没有只看成功率，而是区分 wrong click、筛选后候选选错和 invalid tool call。然后设计了包含格式、专家路径、最终答案、步数和非法动作的 shaped reward，对同一任务采样四条轨迹，用组内 reward 均值计算 relative advantage，再用 advantage-weighted loss 和 frozen SFT reference KL 更新 LoRA。
>
> 正式服务器实验中，SFT baseline 是 60.5%，三个 GRPO seed 是 64.0%、63.0% 和 59.5%，均值 62.17%。最佳 run 有 12 个错变对、5 个对变错，但 p=0.143，没有统计显著。因此我的结论不是“RL 已经稳定提升”，而是“有正向信号，但 seed 方差、zero-advantage group 和多条件候选比较仍需解决”。

## 高频问题

### 1. 为什么要自己搭环境？

真实网页会变化，状态难复现，答案和路径也不容易自动验证。自建环境可以精确知道标准答案、专家路径和每次动作是否合法，使训练与错误分析可重复。

### 2. 为什么不用真实浏览器？

第一阶段目标是验证 post-training 方法链路，而不是证明真实网页泛化。确定性环境降低了浏览器渲染、网络波动和网站变化带来的噪声。真实浏览器是后续扩展，而不是当前实验的替代品。

### 3. SFT 到底学到了什么？

最直接的是工具协议：Base first-step format accuracy 为 0%，SFT 后为 100%。其次是基本页面实体和点击路径映射。完整任务仍只有约 60%，说明格式对齐和决策能力是两个不同阶段。

### 4. 为什么要把轨迹拆成 next-action samples？

每个 assistant turn 都对应一个明确状态和目标动作。拆分后能监督 open、click、submit 等不同阶段，而不是只用整条轨迹做单个长序列样本，也能增加有效训练样本数。

### 5. Reward 为什么不是只有最终答案？

只有 final-answer reward 太稀疏。格式和路径 reward 能区分完全无效、部分路径正确、最终候选错误和完整成功等不同轨迹质量。但路径 reward 也可能强化模仿而不是比较推理，这在 highest-rating 模板退化中得到了体现。

### 6. 你的算法是真正完整的 GRPO 吗？

这是一个 GRPO-style 简化实现：有 grouped sampling、group-relative advantage、advantage-weighted policy update 和 reference KL，但没有完整 PPO ratio clipping。因此应称为 minimal GRPO-KL trainer，而不是宣称完全复现某个工业实现。

### 7. KL 有什么作用？

Reference policy 是训练前的 SFT adapter。KL 惩罚限制 RL policy 不要因为少量高方差 reward 快速偏离已经学会的工具格式和基础策略。早期无 KL prototype 从 70% 降到 45%，说明稳定约束是必要方向。

### 8. 为什么 KL 日志曾出现负数？

k3 estimator 理论上非负，约 `-1e-5` 的值来自 FP16 下 `exp(x)-x-1` 的消减误差。后来改成 FP32 计算并 clamp 到非负。它是数值精度问题，不是 KL 理论为负。

### 9. 为什么 best GRPO 提升，但平均提升不大？

只有 40% group 有非零 advantage，且 batch size 1 的 100-step run 每个 seed 会看到不同样本顺序。Reward 本身也有噪声，所以训练 seed 方差明显。三个 seed 中两个提升、一个退化，平均仅 +1.67 pp。

### 10. 结果显著吗？

不显著。Best seed 的 McNemar exact p=0.143。正确表述是“单次最佳提升 3.5 pp，多 seed 平均方向为正，但未证明稳定显著提升”。

### 11. 为什么还要报告 seed 29 的退化？

只报告最好 seed 会夸大方法效果。退化 seed 能揭示训练方差，也直接指导下一步改善 rollout 多样性、batch coverage、early stopping 和 reward。

### 12. Full-pass 为什么没有更好？

完整覆盖 506 个 action examples 后，成功率仍为 64%，但 invalid calls 增加。固定离线数据上的更多 exposure 带来了 policy drift，没有新增有效信息。这说明训练步数需要通过验证集或 KL/行为指标选择。

### 13. 最大的失败模式是什么？

当前最大问题是多候选选择，特别是排序、筛选和 department + highest-rating 组合。GRPO 对直接标题/名称定位有帮助，但没有稳定改善候选比较。

### 14. 如何避免数据泄漏？

SFT 和 GRPO rollout 都来自 train split，最终指标只来自 eval split。早期脚本曾误用 eval tasks 采样，我发现后废弃该批训练用途并从 train split 重新采样，最终报告只使用修正后的实验。

### 15. 为什么选择 0.5B？

目标是快速验证完整训练闭环和实验设计。0.5B 能在本地完成开发，在单张消费级 GPU 上做多 seed 正式实验。等环境和 reward 更成熟后再扩到 1.5B，比一开始用大模型更节省迭代成本。

### 16. 本地和服务器如何分工？

本地负责环境、代码、单测、smoke 和分析；RTX 5090 服务器负责 400 条 grouped rollout、GRPO 多 seed 训练和多轮 200-task eval。这样昂贵算力只用于真正耗时的生成和训练。

### 17. 如果再做一周，优先改什么？

页面结构、随机 ID、held-out layout 和 targeted tasks 已在 V2 完成。下一步先训练 V2 SFT，确认模型能否根据 observation 复制未见 ID；再根据 15 个模板的失败分布改 reward，之后才提高 group 多样性或换 1.5B。

### 18. V2 相比 V1 最关键的变化是什么？

V1 的 ID 固定且 observation 不显示 ID，模型可能记住点击映射。V2 使用随机无语义 ID，并在每次 observation 显式显示可点击 ID；train 用布局 A/B，eval 用未见布局 C，且 train/eval ID overlap 为 0。因此 V2 更直接测试 observation grounding 和结构泛化。

## 简历表述

### 保守版

- Built a deterministic web-navigation agent environment with synthetic pages, 1,000 verifiable tasks, expert trajectories, strict tool-call parsing, and resumable multi-step evaluation.
- Fine-tuned Qwen2.5-0.5B with LoRA SFT on 2,530 next-action samples, improving first-step tool-call format accuracy from 0% to 100%.
- Implemented shaped rewards, grouped rollouts, group-relative advantages, frozen-reference KL training, failure taxonomy, and paired multi-seed evaluation; observed a best 200-task success rate of 64.0% versus a 60.5% SFT baseline.

### 强调研究严谨性版

- Evaluated GRPO-KL across three training seeds on a held-out 200-task split: best 64.0%, mean 62.17% +/- 2.36 pp versus a 60.5% SFT baseline; reported paired McNemar tests and seed variance rather than only the best run.
- Identified that gains came primarily from direct click-path correction, while filtered/ranked candidate selection remained unchanged or regressed, guiding the next data and reward iteration.

## 不要这样说

- 不要说“GRPO 稳定提升 3.5%”。只有 best seed 是 +3.5 pp。
- 不要说“结果统计显著”。p=0.143。
- 不要说“完整复现了工业 GRPO”。当前是简化 trainer。
- 不要把本地 63.5% SFT 和服务器 64.0% GRPO 直接比较；正式对照是同服务器的 60.5% vs 64.0%。
- 不要隐藏 seed 29 的 59.5%。诚实解释方差反而更能体现实验判断。
