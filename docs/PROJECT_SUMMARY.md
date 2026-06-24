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

这里的 63.5% 来自较早的本地 SFT run。后续正式 GRPO 对比在服务器上重新训练并固定同一 checkpoint，受控 baseline 是 60.5%；不能把本地 63.5% 和服务器 GRPO 64.0% 直接相减。

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

## 7. GRPO 正式实验

已经完成：

- `training/grpo_rollout.py`：对同一道任务采样多条回答。
- 对每条 rollout 计算 reward。
- 在组内计算 advantage：单条 reward 减去同组平均 reward。
- 输出 group rollout 数据，作为 GRPO 训练输入。

正式训练集 group rollout：

- 100 个训练任务。
- 每题 4 个样本。
- 总共 400 条 rollout。
- mean reward：0.5925。
- success rate：56.25%。
- 40/100 个 group 有非零 advantage。
- 生成 506 个 nonzero-advantage action examples。

这说明：

> 同一个模型在 temperature 采样下会产生好坏不同的轨迹，因此组内相对优势信号是存在的。但 60% group 没有 reward 差异，说明当前任务或采样多样性仍是训练效率瓶颈。

## 8. GRPO-KL 结果怎么解释

已经实现了 `training/grpo_train.py`：

- 读取 group rollout。
- 过滤 nonzero advantage 样本。
- 对采样动作计算交叉熵。
- 用 advantage 加权 loss。
- 加入 frozen SFT reference policy 和 KL penalty，降低 RL 更新把 SFT policy 拉偏的风险。
- 更新 LoRA adapter。

在同一台服务器、同一个 200-task eval split 上：

| Run | Success rate | Delta vs SFT |
| --- | ---: | ---: |
| SFT baseline | 60.5% | - |
| GRPO-KL seed 7 | 64.0% | +3.5 pp |
| GRPO-KL seed 17 | 63.0% | +2.5 pp |
| GRPO-KL seed 29 | 59.5% | -1.0 pp |

三次 GRPO-KL 平均成功率是 62.17%，sample standard deviation 是 2.36 pp，平均比 SFT 高 1.67 pp。

最诚实、也最专业的说法：

> 最佳单次结果从 60.5% 提升到 64.0%，而且工具格式保持 100%。但三 seed 中有一个退化，平均提升只有 1.67 个百分点，因此我把它看作初步正向信号，而不是稳定显著提升。这也说明 RL post-training 的训练方差和数据效率本身就是需要解决的问题。

seed 7 的配对结果是 12 个错变对、5 个对变错，McNemar p=0.143，没有达到 0.05 显著水平。

模板分析进一步显示：

- `course_title` 平均提升 21.67 pp。
- `shopping_name` 平均提升 9.88 pp。
- `course_department_highest_rating` 平均下降 36.36 pp。
- 排序/筛选后的候选比较总体没有改善。

## 9. 面试时的完整叙述顺序

推荐按这个顺序讲：

1. 我想做小模型网页导航，所以先搭了一个可验证本地环境。
2. 环境能生成页面、任务、专家路径和标准答案。
3. 我用专家轨迹构造 SFT 数据，训练 Qwen2.5-0.5B 的 LoRA adapter。
4. Base model 完全不遵守工具格式，SFT 后格式准确率到 100%。
5. 本地 SFT run 的 200 题成功率为 63.5%，说明模型已经具备基本导航能力；正式 RL 对比使用同服务器的 60.5% baseline。
6. 错误分析显示主要问题从格式转向候选选择和点击决策。
7. 因此我设计了包含格式、路径、答案和非法动作惩罚的 reward。
8. 再做 group rollout 和 advantage，为 GRPO 做准备。
9. 正式 GRPO-KL 最佳单次提升 3.5 pp，多 seed 平均提升 1.67 pp，但存在明显方差。
10. 下一步不是盲目加训练步数，而是增加有效 advantage group，并针对候选比较改数据和 reward。

## 10. 如果面试官问“你的贡献是什么”

可以回答：

> 我的贡献主要有三块。第一，我搭了一个可复现的本地网页导航环境和任务生成器，让 agent 训练可以稳定验证。第二，我完成了从专家轨迹到 LoRA SFT 的训练和完整 rollout 评测，把 base model 从不会工具调用提升到 100% 格式准确率。第三，我实现了 reward、group rollout、reference KL 和多 seed 配对分析；最佳 GRPO-KL 从同环境 SFT 的 60.5% 提升到 64.0%，同时我也如实分析了 seed 方差和不显著性。

## 11. 如果面试官问“为什么不用服务器”

可以回答：

> 环境、SFT smoke、reward 和代码验证都在本地完成；服务器只用于 400 条 grouped rollout、多 seed GRPO-KL 和多轮 200-task eval。这样把昂贵算力留给生成和正式评测，日常开发仍在本地，成本和迭代效率更平衡。

## 12. 如果面试官问“下一步怎么提高”

可以从四个方向回答：

- 数据：增加任务模板、页面结构和干扰项，减少对固定元素 ID 的记忆。
- 模型：继续做 targeted SFT，重点补排序、筛选、多条件候选选择。
- Reward：用当前 reward 做 rejection sampling 或 reward-weighted SFT。
- RL：提高 group 内采样差异、加入 early stopping，并减少训练 seed 方差。

## 13. 当前一句话结论

这个项目已经完成了环境、数据、SFT、评测、错误分析、reward、group rollout、GRPO-KL 和多 seed 统计分析的完整闭环。受控服务器实验中，SFT 为 60.5%，最佳 GRPO-KL 为 64.0%，三 seed 平均为 62.17%；工具格式始终保持 100%。下一步重点是扩大任务结构多样性、改善候选比较，并降低 RL seed 方差。

## 14. V2 环境升级

针对 V1 固定 element ID、布局单一和候选比较不足的问题，V2 已完成：

- 24 个商品和 24 门课程。
- 210 个页面。
- train A/B 两种已见布局，eval C 为结构级 held-out 布局。
- element ID 使用随机 token，并在 observation 中显式展示。
- train/eval expert element ID overlap 为 0。
- 15 个模板均衡采样，train 每类 200 条。
- 3000 train + 500 eval，共 3500 条任务。
- hard tasks 2102 条。
- expert 3500/3500 成功，非法动作 0。
- 3500 条 expert 路径最终页面全部与 target 匹配。
- 生成 11200 个 train next-action examples。

V2 SFT step1400 已完成，在 held-out layout C 的 500 tasks 上：

- task success rate：31.2%。
- tool-call format accuracy：99.95%。
- easy success：78.0%。
- medium success：19.7%。
- hard success：13.2%。
- correct filter rate：95.9%。
- candidate accuracy after correct filter：15.0%。

Direct lookup 已经能够泛化：shopping name 97.0%，course title 93.9%。但 300 个失败属于 `wrong_candidate_after_filter`，四个最高/最低比较模板为 0%。模型在不同模板中集中选择固定候选位置，例如 under-$100 两个模板始终选择 position 2，说明它仍在利用训练布局中的位置捷径。

因此当时没有直接做 V2 GRPO，而是先设计 V2.1：为同一任务生成大量 candidate shuffle 和随机 ID 页面实例，让位置与答案去相关，再训练 targeted SFT。

## 15. V2.1 候选位置去偏数据

V2.1 数据阶段已经完成：

- 20 个训练页面实例和 5 个独立 grid 评测实例。
- 1750 个页面、4475 个随机 element ID，train/eval ID overlap 为 0。
- 6000 train + 1000 eval，15 个模板在 train 中各 400 条。
- 22400 个 train next-action examples。
- expert 7000/7000 成功，target path 7000/7000 匹配，非法动作和环境错误均为 0。
- 4400 条筛选型训练任务中，任一模板的最大单位置占比为 26%。
- 同一 template-answer 至少覆盖 4 个候选位置，平均覆盖 5.35 个位置。

这里使用 cyclic rotation 而不是只依赖随机 shuffle，使候选位置覆盖可控且可审计。同一个答案在属性不变的情况下出现在不同位置，构成 counterfactual training pair，迫使模型减少对固定位置的依赖。

实现时还发现 20 个实例和 15 个模板按全局索引分配会因 `gcd(20, 15) = 5` 导致每个模板只覆盖 4 个实例。最终改成按模板自身出现次数轮转 context，并在单测中断言每个模板覆盖全部训练实例。

当前尚未得到 V2.1 模型结果。下一步在服务器上用 22400 条 action examples 训练一轮 SFT；effective batch size 为 8 时约 2800 optimizer steps。评测重点不是只看总成功率，而是看 ranking 模板、correct-filter 后 candidate accuracy 和不同候选位置上的鲁棒性。
