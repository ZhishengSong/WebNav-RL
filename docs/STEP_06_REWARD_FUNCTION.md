# Step 06: Reward Function and Breakdown Analysis

本步骤实现了第一版 reward function，并在已有的 200 条 SFT rollout 上验证 reward 是否能合理区分成功、路径部分正确、候选错误和无效工具调用。

核心结论：

```text
success mean reward: 0.8764
failure mean reward: 0.2212
wrong_candidate_after_filter mean reward: 0.35
wrong_click_path mean reward: 0.20
invalid_tool_call mean reward: -0.0875
```

这说明 reward 设计基本符合预期：成功样本高分，路径部分正确但答案错的样本中等分，完全点错路径的样本低分，无效工具调用会被明显惩罚。

## 1. 为什么现在做 reward

前面已经完成：

```text
Base eval
LoRA SFT
完整 200 条 SFT eval
error analysis
```

SFT step200 的结果是：

```text
success rate: 63.5%
tool-call format accuracy: 100%
invalid tool-call rate: 1.88%
```

也就是说，模型已经稳定进入可交互工具调用空间。现在做 reward 的目的不是立刻训练 GRPO，而是先回答：

- 成功轨迹是否能拿到高 reward？
- 失败轨迹是否会被扣分？
- 选对 filter 但选错候选，是否能拿到部分 path reward？
- invalid tool call 是否会被惩罚？
- 这个 reward 是否适合后续接 GRPO？

## 2. 新增文件

```text
training/reward.py
```

功能：

- 读取 rollout trajectories。
- 读取对应 task metadata。
- 计算每条 trajectory 的 reward breakdown。
- 输出总报告和逐条 breakdown。

运行命令：

```powershell
python training\reward.py `
  --trajectories outputs\trajectories\sft_qwen_0_5b_step200_eval200_full_t48_trajectories.jsonl `
  --tasks tasks\eval_tasks.jsonl `
  --output outputs\eval_reports\sft_qwen_0_5b_step200_eval200_reward_report.json `
  --breakdown outputs\eval_reports\sft_qwen_0_5b_step200_eval200_reward_breakdown.jsonl
```

## 3. Reward 公式

当前 reward 由 6 个部分组成：

```text
total_reward =
  format_reward
  + path_reward
  + answer_reward
  + step_penalty
  + invalid_penalty
  + no_submit_penalty
```

具体定义：

```text
format_reward:
  +0.2 if all model outputs parse as valid tool calls

path_reward:
  +0.3 * prefix_match(model_clicks, expert_clicks)

answer_reward:
  +0.4 if final answer exact matches target

step_penalty:
  -0.05 per model step beyond expert_clicks + open + submit

invalid_penalty:
  -0.2 per invalid tool call

no_submit_penalty:
  -0.2 if episode does not submit
```

### 为什么这样设计

`format_reward` 鼓励模型遵守工具协议。没有这个项，RL 早期很容易浪费在 parser error 上。

`path_reward` 给部分正确行为中间奖励。比如模型先点对 `filter_credits_4`，但后面候选选错，它仍能拿到部分分数。

`answer_reward` 是最终任务目标，权重最高之一。

`step_penalty` 鼓励短路径，避免循环点击。

`invalid_penalty` 明确惩罚不存在的工具、错误参数或不存在的 element_id。

`no_submit_penalty` 防止模型一直操作但不结束。

## 4. Path reward 设计

path reward 使用 task 中的 expert path：

```json
"expert_clicks": ["filter_credits_4", "course_credits_4_item_001"]
```

模型执行后可以得到：

```json
"model_clicks": ["filter_credits_4", "course_credits_4_item_002"]
```

然后计算 prefix match：

```text
model_clicks 和 expert_clicks 从头开始匹配
匹配到第一个不同为止
path_score = matched_prefix_len / len(expert_clicks)
path_reward = 0.3 * path_score
```

例子：

```text
expert_clicks = [filter, correct_item]
model_clicks  = [filter, wrong_item]
path_score = 1 / 2 = 0.5
path_reward = 0.15
```

这个设计能区分：

- 完全没走对路径：0
- 过滤/排序对了但候选错了：中间分
- 完整路径正确：满分 0.3

## 5. 输出文件

Reward report：

```text
outputs/eval_reports/sft_qwen_0_5b_step200_eval200_reward_report.json
```

逐条 breakdown：

```text
outputs/eval_reports/sft_qwen_0_5b_step200_eval200_reward_breakdown.jsonl
```

每条 breakdown 示例：

```json
{
  "task_id": "shop_00801",
  "success": true,
  "format_reward": 0.2,
  "path_reward": 0.3,
  "path_score": 1.0,
  "answer_reward": 0.4,
  "step_penalty": -0.0,
  "invalid_penalty": -0.0,
  "no_submit_penalty": 0.0,
  "total_reward": 0.9,
  "model_clicks": ["shop_item_007"],
  "expert_clicks": ["shop_item_007"],
  "final_answer": "PulseBand Neo",
  "target_answer": "PulseBand Neo"
}
```

## 6. Reward report 结果

完整统计：

```json
{
  "num_rewards": 200,
  "mean_total_reward": 0.63725,
  "min_total_reward": -0.25,
  "max_total_reward": 0.9,
  "success_count": 127,
  "failure_count": 73,
  "success_mean_reward": 0.8763779527559056,
  "failure_mean_reward": 0.22123287671232877,
  "mean_format_reward": 0.2,
  "mean_path_reward": 0.19575,
  "mean_answer_reward": 0.254,
  "mean_step_penalty": -0.0005,
  "mean_invalid_penalty": -0.012
}
```

解释：

- 成功样本平均 reward 接近 0.9。
- 失败样本平均 reward 约 0.22。
- mean format reward 是 0.2，说明所有样本都符合格式。
- mean answer reward 是 0.254，对应 127/200 成功率。
- step penalty 非常小，说明模型路径长度没有明显膨胀。
- invalid penalty 很小但非零，对应少量 invalid tool call。

## 7. 按错误类型看 reward

```json
{
  "invalid_tool_call": {
    "count": 4,
    "mean_reward": -0.0875
  },
  "success": {
    "count": 127,
    "mean_reward": 0.8764
  },
  "wrong_candidate_after_filter": {
    "count": 18,
    "mean_reward": 0.35
  },
  "wrong_click_path": {
    "count": 51,
    "mean_reward": 0.2
  }
}
```

这组结果很符合设计直觉：

- `success` 高分。
- `wrong_candidate_after_filter` 有部分路径正确，所以是中间分 0.35。
- `wrong_click_path` 通常只有格式分，所以约 0.2。
- `invalid_tool_call` 因为无效动作惩罚，平均分为负。

## 8. 按 template 看 reward

表现较好的模板：

```text
course_department_time: 0.90
course_4_credit_department: 0.81
course_code: 0.77
course_title: 0.74
```

表现较弱的模板：

```text
course_4_credit_highest_rating: 0.35
shopping_category_highest_rating: 0.39
course_department_highest_rating: 0.44
shopping_name: 0.49
```

这和 error analysis 一致：涉及 highest rating + category/department 的任务最容易出错，因为它需要在候选里做条件约束下的比较，而不是直接选全局最高。

## 9. 面试时怎么讲

可以这样讲：

> 在 SFT 模型能稳定调用工具后，我设计了一个 rule-based reward function。它不是只看最终答案，而是拆成 format reward、path reward、answer reward、step penalty、invalid penalty 和 no-submit penalty。这样成功样本会得到高分，路径部分正确但候选错的样本会得到中间分，无效工具调用会被强惩罚。我在 200 条 SFT rollout 上验证，成功样本平均 reward 是 0.876，失败样本是 0.221，invalid tool call 平均是负分。这说明 reward 能较好地区分不同质量的轨迹，可以作为后续 GRPO 的基础。

如果被问为什么需要 path reward：

> 只用 final answer reward 太稀疏。比如模型先点对了过滤按钮，但候选选错，如果只看最终答案就是 0；但这种轨迹其实比完全乱点更接近正确行为。path reward 给它部分信用，有利于 RL 在多步任务里学习中间决策。

如果被问这个 reward 有没有风险：

> 有。path reward 依赖 expert path，可能会偏向单一路径；如果一个任务存在多条等价路径，这种 reward 会低估其他路径。所以当前环境里先用 deterministic expert path，后续如果扩展任务，需要支持多参考路径或更语义化的 verifier。

## 10. 当前局限

当前 reward 还有几个局限：

- path reward 依赖 `expert_clicks`，不支持多条正确路径。
- answer reward 是 exact match，对别名/同义答案不宽容。
- 没有单独奖励“候选集合正确但 final answer 格式错”的情况。
- step penalty 很简单，没有区分必要探索和无效循环。
- 还没有接到 GRPO trainer，只是在离线 trajectories 上验证。

这些都是后续可以改进的点。

## 11. 下一步

现在可以进入 GRPO 前的准备：

1. 写 rollout group sampler：
   ```text
   同一个 task 采样 group_size=4 条 trajectory
   ```
2. 用 `training/reward.py` 给每条 trajectory 算 reward。
3. 计算 group-relative advantage。
4. 先做一个小规模 GRPO prototype。

也可以先做 reward ablation：

```text
full reward
only final answer reward
no path reward
no step penalty
```

但从当前进度看，最自然的下一步是：

```text
GRPO rollout data structure + group reward computation
```
