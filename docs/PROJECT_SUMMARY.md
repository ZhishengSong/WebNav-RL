# WebNav-RL 项目整理与面试讲解稿

这份文档把目前已经完成的工作串成一条主线，目标是让你面试时能讲清楚：项目为什么做、系统怎么搭、模型怎么训、指标说明什么、下一步为什么要做 RL。

## 1. 项目一句话

WebNav-RL 是一个本地可验证的网页导航智能体训练项目。我搭了一个小型浏览器环境，自动生成购物和课程网页，再生成查找类任务，让模型通过 `open_page`、`click`、`get_visible_text`、`submit_answer` 这些工具完成网页导航。整个项目覆盖了从环境构建、专家轨迹、SFT 训练、模型评测、错误分析、奖励函数到 GRPO 原型的完整链路。

面试时可以这样开场：

> 我做的是一个小模型网页导航 post-training 项目。不是只做静态问答，而是让模型在本地网页环境里一步步调用工具、点击元素、读取页面并提交答案。这个环境是可控和可验证的，所以可以系统地做 SFT、评测、错误分析和后续 RL。

## 2. 为什么要自己搭环境

真实网页任务很难直接训练和评测，因为网页会变、状态复杂、答案也不好稳定验证。这个项目先把问题缩小成一个可控版本：

- 页面由程序生成，所以结构、元素 ID、答案都可追踪。
- 任务由 metadata 生成，所以每道题都有标准答案和专家路径。
- 工具调用是固定格式，所以可以严格统计格式错误、非法工具、成功率。
- 所有流程都能在本地电脑跑，不依赖远程服务。

面试要点：

> 我没有一上来就追求真实网页规模，而是先构建了一个可复现实验闭环。这样每次模型失败时，我能知道它是格式错、点错路径、候选选错，还是最终答案错。

## 3. 已完成的系统模块

### 环境与任务

- `pages/page_generator.py` 生成购物页和课程页。
- `tasks/task_generator.py` 生成 1000 条任务，划分为 800 训练、200 评测。
- `env/browser_env.py` 实现工具式环境交互。
- `env/verifier.py` 做最终答案 exact match 验证。
- `rollout/rollout_runner.py` 跑规则专家，生成专家轨迹。

这一步产出了：

- `training/sft_train.jsonl`：800 条 SFT 训练数据。
- `training/sft_eval.jsonl`：200 条 SFT 评测数据。
- 专家轨迹文件：用于训练、对齐和 reward 里的路径对比。

### 模型评测

- `rollout/parser.py` 解析模型输出里的 `<tool_call>{...}</tool_call>`。
- `rollout/model_runner.py` 负责模型输出、工具执行、环境状态更新。
- `scripts/run_eval.py` 支持 base model、LoRA adapter、断点续跑、增量保存。

这一步很关键，因为它把模型从“输出文本”接进了真实工具循环。

### LoRA SFT

- `training/sft_train.py` 用专家轨迹构造成 next-action 训练样本。
- 使用 Qwen2.5-0.5B-Instruct 作为 base model。
- 训练了 `outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200`。

训练规模：

- 800 条专家轨迹。
- 2530 个 next-action 样本。
- 200 个 optimizer steps。
- LoRA 可训练参数约 4.4M，占总参数约 0.88%。

## 4. 核心结果怎么讲

### Base model 结果

Base Qwen 在 20 道 first-step eval 上：

- tool-call format accuracy：0.0
- invalid tool-call rate：1.0
- success：0

解释：

> 这说明通用 chat model 不会自然遵守我定义的工具调用协议。它不是不会语言理解，而是没有被对齐到这个环境的 action format。

### SFT 后结果

SFT step200 在 200 道 first-step eval 上：

- tool-call format accuracy：1.0
- invalid tool-call rate：0.0

完整 200 题 rollout：

- success：127/200
- success rate：63.5%
- tool-call format accuracy：1.0
- invalid tool-call rate：1.88%
- average model steps：3.195

解释：

> SFT 首先解决的是“会不会按协议行动”的问题。Base model 连合法工具调用都不稳定，SFT 后格式准确率达到 100%。在完整多步任务里，模型能完成 63.5% 的 held-out 任务，说明它已经学到了一部分网页导航和点击策略。

## 5. 错误分析怎么讲

完整评测失败 73 条，主要分为：

- `wrong_click_path`：51 条。
- `wrong_candidate_after_filter`：18 条。
- `invalid_tool_call`：4 条。

这个结果说明：

- 格式已经不是主要问题。
- 主要难点变成了页面理解和候选项选择。
- 特别是排序、筛选、多条件任务下，模型容易选择列表里的错误候选。

面试时可以这样说：

> 我没有只看总成功率，而是把失败按行为类型拆开。SFT 后格式问题基本解决，剩下的问题集中在决策质量：比如筛选后应该点第几个候选、排序后最高评分到底是哪一个。这也指导了下一步不能只继续做格式 SFT，而应该引入 reward 或更有针对性的数据。

## 6. 奖励函数设计

`training/reward.py` 里做了一个 shaped reward：

- 输出格式合法：+0.2
- 点击路径前缀匹配专家：最多 +0.3
- 最终答案正确：+0.4
- 多余步数惩罚：每步 -0.05
- 非法工具惩罚：每次 -0.2
- 没有 submit 惩罚：-0.2

在 SFT 200-task 轨迹上：

- 平均 reward：0.637
- 成功样本平均 reward：0.876
- 失败样本平均 reward：0.221
- `wrong_candidate_after_filter` 平均 reward：0.35
- `wrong_click_path` 平均 reward：0.2
- `invalid_tool_call` 平均 reward：-0.0875

解释：

> 这个 reward 不是只看最终答案，而是把格式、过程路径和最终答案结合起来。这样即使模型最后错了，也能区分“格式完全错”“路径部分对但候选错”“最终答案正确”等不同质量的轨迹。

## 7. GRPO 当前进度

已经完成：

- `training/grpo_rollout.py`：对同一道任务采样多条回答。
- 对每条 rollout 计算 reward。
- 在组内计算 advantage：单条 reward 减去同组平均 reward。
- 输出 group rollout 数据，作为 GRPO 训练输入。

真实 SFT 模型的小规模 group rollout：

- 4 个任务。
- 每题 4 个样本。
- 总共 16 条 rollout。
- mean reward：0.5656。
- success rate：56.25%。
- 3/4 个 group 有非零 advantage。

这说明：

> 同一个模型在 temperature 采样下会产生好坏不同的轨迹，因此组内相对优势信号是存在的。这是 GRPO 能继续推进的前提。

## 8. Minimal GRPO Prototype 怎么解释

已经实现了 `training/grpo_train.py`：

- 读取 group rollout。
- 过滤 nonzero advantage 样本。
- 对采样动作计算交叉熵。
- 用 advantage 加权 loss。
- 加入 frozen SFT reference policy 和 KL penalty，降低 RL 更新把 SFT policy 拉偏的风险。
- 更新 LoRA adapter。

但是当前 prototype 的结果是：

- 20 题 eval 成功率：45%。
- 同样 20 题上 SFT 约 70%。
- 所以 prototype 发生了退化。

这个要诚实讲：

> 最早的 minimal prototype 我不会把它包装成性能提升。它的意义是证明 RL 数据流和训练流已经打通了：能采样、能算 reward、能算 advantage、能更新 LoRA、能重新评测。它当时退化的原因也比较明确：数据太少，还没有 KL/reference policy，也没有足够 group 规模。

现在 Step 09 已经补上了 reference/KL 机制，并完成了 1-step smoke run。更准确的说法是：

> 目前 GRPO-KL 的训练基础设施已经打通，但还没有做足够规模的性能实验。下一步要扩大 group rollout，并在同一 eval split 上比较 SFT 和 GRPO-KL adapter。

## 9. 面试时的完整叙述顺序

推荐按这个顺序讲：

1. 我想做小模型网页导航，所以先搭了一个可验证本地环境。
2. 环境能生成页面、任务、专家路径和标准答案。
3. 我用专家轨迹构造 SFT 数据，训练 Qwen2.5-0.5B 的 LoRA adapter。
4. Base model 完全不遵守工具格式，SFT 后格式准确率到 100%。
5. 完整 200 题评测成功率 63.5%，说明模型已经具备基本导航能力。
6. 错误分析显示主要问题从格式转向候选选择和点击决策。
7. 因此我设计了包含格式、路径、答案和非法动作惩罚的 reward。
8. 再做 group rollout 和 advantage，为 GRPO 做准备。
9. 当前 GRPO prototype 已打通链路，但还没带来提升，下一步需要 KL、更多 rollouts 和更稳定训练。

## 10. 如果面试官问“你的贡献是什么”

可以回答：

> 我的贡献主要有三块。第一，我搭了一个可复现的本地网页导航环境和任务生成器，让 agent 训练可以稳定评测。第二，我完成了从专家轨迹到 LoRA SFT 的训练和完整 rollout 评测，把 base model 从不会工具调用提升到 100% 格式准确率和 63.5% 任务成功率。第三，我做了失败模式分析和 reward/GRPO 原型，把项目从 SFT 扩展到 RL post-training 的方向。

## 11. 如果面试官问“为什么不用服务器”

可以回答：

> 当前阶段主要是 0.5B 小模型、LoRA、短上下文、本地合成任务，所以本地 RTX 5070 Laptop 足够跑通训练和评测。服务器更适合后续扩大模型、扩大 rollout 数量、做多组采样和正式 GRPO 训练时使用。目前先在本地把方法链路验证清楚，成本更低，也更容易快速迭代。

## 12. 如果面试官问“下一步怎么提高”

可以从四个方向回答：

- 数据：增加任务模板、页面结构和干扰项，减少对固定元素 ID 的记忆。
- 模型：继续做 targeted SFT，重点补排序、筛选、多条件候选选择。
- Reward：用当前 reward 做 rejection sampling 或 reward-weighted SFT。
- RL：给 GRPO 加 KL/reference policy，扩大 group rollout，再和 SFT baseline 做同 split 对比。

## 13. 当前一句话结论

这个项目现在已经完成了从本地环境到 SFT 成功模型、错误分析、reward 设计和 GRPO 原型的完整基础闭环。最强结果是 SFT adapter 在 200 道 held-out 任务上达到 63.5% 成功率和 100% 工具格式准确率；当前最明确的下一步是把 GRPO 从 proof of concept 扩展成稳定、有 KL 约束、更多采样的正式训练实验。
