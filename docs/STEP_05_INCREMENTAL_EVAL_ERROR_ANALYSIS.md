# Step 05: Incremental Eval and Error Analysis

本步骤完成了两个关键改进：

1. 让 eval 支持增量保存和 resume。
2. 对 SFT step200 的完整 200 条 eval 做 error analysis。

核心结果：

```text
SFT step200 full eval on 200 tasks:
success rate: 63.5%
tool-call format accuracy: 100%
invalid tool-call rate: 1.88%
average steps: 3.195
```

这一步把之前的 50 条小评估扩展成了完整 200 条 eval，并明确分析了剩下失败案例主要失败在哪里。

## 1. 为什么要做 incremental/resume

上一阶段尝试跑 200 条完整 rollout 时遇到一个工程问题：

```text
eval 全部跑完才写报告
如果中途超时，已经跑完的结果会丢失
```

真实模型 rollout 比 expert replay 慢很多。一个任务可能要生成多轮 tool call，每轮都要经过模型生成、parser、环境执行和 observation 回填。

因此这一步给 eval 增加：

```text
--incremental
--resume
--report-every
```

现在长评估可以：

- 每完成一条 trajectory 就 append 到 jsonl。
- 每 N 条刷新一次 report。
- 如果中途超时，再运行同一命令会跳过已完成 task。
- 最终 trajectory 文件不会重复 task_id。

## 2. 修改的文件

```text
eval/evaluate.py
scripts/run_eval.py
eval/error_analysis.py
README.md
docs/STEP_05_INCREMENTAL_EVAL_ERROR_ANALYSIS.md
```

### 2.1 `eval/evaluate.py`

新增 helper：

```python
resolve_path(...)
load_jsonl(...)
write_report(...)
append_jsonl(...)
```

`evaluate_tasks(...)` 新增参数：

```python
resume: bool = False
incremental: bool = False
report_every: int = 10
```

核心行为：

```text
如果 resume=True 且 output_path 已存在
-> 读取已有 trajectories
-> 收集已完成 task_id
-> 跳过这些 task
-> 只跑剩余任务
```

如果 `incremental=True`：

```text
每完成一条 -> append 到 output jsonl
每 report_every 条 -> 刷新 report json
```

### 2.2 `scripts/run_eval.py`

新增 CLI 参数：

```text
--resume
--incremental
--report-every
```

metadata 里也会记录：

```json
{
  "resume": true,
  "incremental": true,
  "completed_tasks": 200,
  "pending_tasks": 0
}
```

这样之后看报告时能知道它是否来自可恢复长评估。

## 3. 完整 200 条 eval 命令

使用 SFT step200 adapter：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200
```

命令：

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe scripts\run_eval.py `
  --model models\qwen2.5-0.5b-instruct `
  --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step200 `
  --tasks tasks\eval_tasks.jsonl `
  --limit 200 `
  --output outputs\trajectories\sft_qwen_0_5b_step200_eval200_full_t48_trajectories.jsonl `
  --report outputs\eval_reports\sft_qwen_0_5b_step200_eval200_full_t48_report.json `
  --failures outputs\eval_reports\sft_qwen_0_5b_step200_eval200_full_t48_failures.jsonl `
  --device auto `
  --max-new-tokens 48 `
  --temperature 0.0 `
  --incremental `
  --resume `
  --report-every 10
```

第一次运行在 40 分钟超时前完成并保存了 123 条 trajectory，其中 report 刷新到了 120 条。

第二次运行同一命令后自动 resume，跳过已完成 task，最终完成 200 条。

校验：

```text
trajectory lines: 200
unique task_id: 200
duplicates: 0
```

## 4. 完整 200 条 eval 结果

报告文件：

```text
outputs/eval_reports/sft_qwen_0_5b_step200_eval200_full_t48_report.json
```

结果：

```json
{
  "num_tasks": 200,
  "successes": 127,
  "task_success_rate": 0.635,
  "final_answer_accuracy": 0.635,
  "tool_call_format_accuracy": 1.0,
  "invalid_tool_call_rate": 0.018779342723004695,
  "average_model_steps": 3.195,
  "average_environment_actions": 3.195,
  "format_errors": 0,
  "invalid_tool_calls": 12,
  "termination_counts": {
    "submitted": 200
  }
}
```

解释：

- 200 条 eval 全部跑完。
- 127 条成功，成功率 63.5%。
- tool-call format accuracy 仍然是 100%。
- 没有 parser format error。
- invalid tool call 很低，约 1.88%。
- 所有任务都走到了 `submit_answer`。
- 平均 3.195 步，和 expert trajectory 的长度接近。

这说明 SFT step200 已经是一个比较稳定的 agentic tool-use baseline。

## 5. Error analysis 脚本

新增文件：

```text
eval/error_analysis.py
```

命令：

```powershell
python eval\error_analysis.py `
  --trajectories outputs\trajectories\sft_qwen_0_5b_step200_eval200_full_t48_trajectories.jsonl `
  --tasks tasks\eval_tasks.jsonl `
  --output outputs\eval_reports\sft_qwen_0_5b_step200_eval200_error_analysis.json `
  --examples outputs\eval_reports\sft_qwen_0_5b_step200_eval200_error_examples.jsonl `
  --max-examples-per-type 5
```

输出：

```text
outputs/eval_reports/sft_qwen_0_5b_step200_eval200_error_analysis.json
outputs/eval_reports/sft_qwen_0_5b_step200_eval200_error_examples.jsonl
```

## 6. 错误分类逻辑

当前分类规则是 rule-based：

```text
success
format_error
invalid_tool_call
max_steps_no_submit
missing_final_answer
premature_submit
wrong_candidate_after_filter
wrong_click_path
wrong_final_answer
other_failure
```

最重要的两个分类：

### wrong_click_path

模型点击路径和 expert path 第一处就不同。

例子：

```text
instruction: Find the item named SoundCore A20.
model_clicks:  ['shop_item_002']
expert_clicks: ['shop_item_001']
final_answer: BassFlow Mini
target_answer: SoundCore A20
```

这说明模型基本学会了“点击详情页并提交”，但具体 item 选择错了。

### wrong_candidate_after_filter

模型先选对了 filter/sort，但在过滤后的候选列表中选错候选。

例子：

```text
instruction: Find the highest rated bluetooth earbuds.
model_clicks:  ['sort_rating_desc', 'shop_rating_item_001']
expert_clicks: ['sort_rating_desc', 'shop_rating_item_002']
final_answer: FocusLamp Max
target_answer: SoundCore A20
```

这里模型知道要排序，但错误地选择了排序列表第一项，没有继续考虑 category 条件。

这个错误很关键，因为它说明模型需要更强的条件组合和候选比较能力。

## 7. Error analysis 结果

报告文件：

```text
outputs/eval_reports/sft_qwen_0_5b_step200_eval200_error_analysis.json
```

结果：

```json
{
  "num_trajectories": 200,
  "successes": 127,
  "failures": 73,
  "success_rate": 0.635,
  "failure_rate": 0.365,
  "error_counts": {
    "success": 127,
    "wrong_candidate_after_filter": 18,
    "wrong_click_path": 51,
    "invalid_tool_call": 4
  }
}
```

错误占比：

```text
wrong_click_path: 25.5%
wrong_candidate_after_filter: 9.0%
invalid_tool_call: 2.0%
```

在所有失败的 73 条里：

```text
wrong_click_path: 51 / 73 = 69.9%
wrong_candidate_after_filter: 18 / 73 = 24.7%
invalid_tool_call: 4 / 73 = 5.5%
```

结论：

> 当前主要瓶颈不是工具格式，而是候选选择和多条件筛选。

## 8. 按页面类型拆分

```json
{
  "shopping": {
    "success": 51,
    "wrong_candidate_after_filter": 12,
    "wrong_click_path": 41,
    "invalid_tool_call": 2
  },
  "course": {
    "success": 76,
    "wrong_candidate_after_filter": 6,
    "invalid_tool_call": 2,
    "wrong_click_path": 10
  }
}
```

解释：

- course 任务明显更容易，成功 76 条。
- shopping 失败更多，尤其是 wrong_click_path。
- shopping 中颜色、价格、类别、评分组合更容易混淆。

## 9. 按难度拆分

```json
{
  "easy": {
    "success": 98,
    "wrong_click_path": 43,
    "invalid_tool_call": 1
  },
  "medium": {
    "wrong_candidate_after_filter": 18,
    "success": 29,
    "invalid_tool_call": 3,
    "wrong_click_path": 8
  }
}
```

解释：

- easy 任务仍有不少 wrong_click_path，说明模型在直接 item lookup 上还有记忆/定位错误。
- medium 任务主要是 wrong_candidate_after_filter，说明模型会用 filter/sort，但组合条件判断不足。

## 10. 按 template 拆分

失败比较集中的模板：

```text
shopping_category_highest_rating:
  wrong_candidate_after_filter: 12
  success: 1

shopping_name:
  success: 12
  wrong_click_path: 15

shopping_price_lookup:
  success: 13
  wrong_click_path: 13
  invalid_tool_call: 1

course_4_credit_highest_rating:
  wrong_candidate_after_filter: 5
```

这说明下一步最值得增强的数据/训练方向是：

- 最高评分 + 类别条件。
- 商品名直接定位。
- 价格直接定位。
- 4 credit + highest rating 的候选比较。

## 11. 面试时怎么讲

可以这样讲：

> 我发现完整 200 条 rollout 时间比较长，所以先给 eval pipeline 加了 incremental save 和 resume。这样每跑完一条 trajectory 就写入 jsonl，并且定期刷新 report；如果超时，再运行同一个命令会跳过已完成 task。用这个机制我跑完了 SFT 模型的 200 条完整 eval，成功率 63.5%，tool-call format accuracy 100%，invalid tool-call rate 约 1.88%。随后我做了 rule-based error analysis，发现失败主要不是格式错误，而是 wrong_click_path 和 wrong_candidate_after_filter，说明 SFT 已经解决了工具协议问题，下一步要提升的是候选选择和多条件比较能力。

如果被问“error analysis 对后续有什么指导”：

> 它直接告诉我后续数据和 reward 怎么设计。比如 shopping_category_highest_rating 很容易选成全局最高评分商品，而不是指定类别里的最高评分商品；这说明后续可以生成更多需要 category-constrained ranking 的任务，也可以在 reward 里加入 path/candidate reward，惩罚选错过滤后候选的行为。

## 12. 当前结论

当前项目已经完成：

```text
本地 WebNav environment
expert trajectory generation
SFT data construction
Base model eval
LoRA SFT
full 200-task SFT eval with resume
error analysis
```

目前可以比较有把握地说：

```text
Base model 完全不会工具协议
SFT 后工具协议稳定到 100%
SFT full rollout 成功率达到 63.5%
主要失败来自候选选择，而不是格式错误
```

## 13. 下一步建议

下一步可以有两个方向：

### 方向 A：继续强化 SFT

基于 error analysis 生成更多失败类型数据：

```text
shopping_category_highest_rating
shopping_price_lookup
shopping_name
course_4_credit_highest_rating
```

然后再训练一版 SFT。

### 方向 B：进入 Reward / GRPO

现在 SFT 已经有稳定 tool-call 协议，可以进入 RL 阶段：

```text
format reward
final answer reward
invalid action penalty
step penalty
path/candidate reward
```

推荐先做：

```text
reward.py + reward breakdown
```

然后用已有 trajectories 测 reward，再接 GRPO。
