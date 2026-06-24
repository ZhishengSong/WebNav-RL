# Step 13: V2.1 Candidate-Shuffled Data

本步骤针对 V2 SFT 暴露出的候选位置捷径，构造多页面实例、随机 element ID、候选循环换位的数据集。目标不是直接提高指标，而是先消除训练数据中“答案和固定位置相关”这一结构性缺陷。

## 1. 为什么需要 V2.1

V2 SFT 在 held-out layout C 上的工具格式准确率为 99.95%，筛选控件准确率为 95.9%，但正确筛选后的候选准确率只有 15.0%。300 个失败属于 `wrong_candidate_after_filter`，多个最高/最低排序模板成功率为 0。

行为分析进一步发现，模型不是读取 rating、price、credits 等属性后比较，而是在不同模板中反复点击固定位置。例如 under-$100 的两个模板始终选择 position 2。这说明 V2 虽然解决了固定 element ID 泄漏，却仍保留了 candidate order shortcut。

因此修复顺序是：

```text
固定候选顺序的 V2 SFT
-> 定位 position shortcut
-> 构造同答案、不同位置的 counterfactual 页面实例
-> 重新训练 V2.1 SFT
-> 再判断是否值得进入 GRPO
```

## 2. 数据设计

V2.1 保留 V2 的 24 个商品、24 门课程和 15 个任务模板，但扩展为：

- 20 个训练页面实例。
- 5 个独立评测页面实例。
- 每个实例 70 个页面、179 个随机 element ID。
- train 使用 list/compact 两种展示样式交替出现。
- eval 使用未见的 grid 样式。
- 每个实例重新生成无语义 element ID。
- 候选列表按实例编号做 cyclic rotation。
- 同一模板按自身出现次数轮流分配到所有实例。

循环换位保留候选集合和正确答案，只改变候选在 observation 中的位置。因此它构成一种受控 counterfactual：如果模型真正依据属性比较，换位不应改变答案；如果模型依赖位置，换位会迫使训练 loss 纠正这种策略。

## 3. 为什么使用循环换位

纯随机 shuffle 可以增加多样性，但有限样本下可能仍然不均衡。循环换位能给出更可控的覆盖：对一个 4 候选列表，连续实例会让每个候选依次出现在 position 1、2、3、4。

实现入口：

```text
pages/v2_generator.py       通用 rotation_index 支持
pages/v21_generator.py      多实例页面和独立 ID 生成
tasks/v2_task_generator.py  跨上下文模板分配
scripts/run_v21_data.py     生成、审计、expert replay、SFT 构造
```

V2 默认参数不启用换位，所以旧数据生成行为保持不变；V2.1 显式传入新参数。

## 4. 发现并修复的隐藏相关性

第一版实现按全局 task index 同时选择模板和页面实例。20 个训练实例、15 个模板的最大公约数是 5，因此某个固定模板实际上只会落到 4 个实例，而不是 20 个实例。

这类问题很隐蔽：总任务数和模板计数都正确，但模板到上下文的联合分布不完整。修复方法是使用“该模板已经出现了多少次”选择 context：

```text
template_occurrence = index // number_of_templates
context = contexts[template_occurrence % number_of_contexts]
```

测试现在明确断言每个训练模板都覆盖全部训练实例，防止以后再次出现模数耦合。

## 5. 完整数据结果

生成命令：

```bash
python scripts/run_v21_data.py \
  --train-tasks 6000 \
  --eval-tasks 1000 \
  --train-instances 20 \
  --eval-instances 5 \
  --seed 71 \
  --report outputs/eval_reports/v21_data_report.json
```

核心结果：

| Metric | Result |
| --- | ---: |
| Pages | 1,750 |
| Element IDs | 4,475 |
| Train tasks | 6,000 |
| Eval tasks | 1,000 |
| Templates | 15 |
| Train tasks per template | 400 |
| Expert successes | 7,000/7,000 |
| Expert path target matches | 7,000/7,000 |
| Invalid actions | 0 |
| Action errors | 0 |
| Train/eval ID overlap | 0 |
| Train next-action examples | 22,400 |
| Eval next-action examples | 3,732 |

## 6. 位置去偏审计

对所有包含筛选后候选点击的训练任务，记录最终正确候选在页面中的位置：

```text
filtered train tasks: 4400
filtered templates: 11
maximum single-position share within a template: 26.0%
minimum unique positions for the same template-answer pair: 4
mean unique positions for a template-answer pair: 5.35
```

这三个指标分别回答：

1. 某个模板是否仍集中在固定位置。
2. 同一个正确答案是否真的出现在多个位置。
3. counterfactual 覆盖是否只有最低限度，还是有更广泛变化。

生成脚本设置硬门槛：训练集任一模板的最大位置占比不能超过 40%，同一 template-answer 至少覆盖 2 个位置。最终数据明显优于门槛：最大 26%，最少 4 个位置。

eval 的 5 个独立实例也做相同审计，但不使用训练门槛，因为 eval 的职责是测量泛化，不参与优化。

## 7. 完整性验证

验证分为三层：

1. 生成时检查 page ID 和 element ID 不重复，train/eval ID overlap 为 0。
2. 对 7,000 条任务运行规则专家，检查路径、最终答案、非法动作和环境错误。
3. 通过通用 `scripts/run_eval.py` 对 held-out 数据回放 50 条任务，结果为 50/50，格式与非法调用均为 0。

回归测试结果：

```text
15 passed, 2 skipped
```

其中 V2.1 测试覆盖多实例生成、模板实例覆盖、位置分布、expert replay，以及已提交数据报告的训练合同。

## 8. 产物

版本控制内保留小型、可审计的代码和报告：

```text
pages/v21_generator.py
scripts/run_v21_data.py
tests/test_v21_data.py
outputs/eval_reports/v21_data_report.json
```

大体积生成数据由 `.gitignore` 排除：

```text
pages/generated_pages_v21/
tasks/v21/
training/v21/
outputs/trajectories/v21_*.jsonl
```

## 9. 下一步训练计划

V2.1 训练集有 22,400 个 next-action examples。effective batch size 为 8 时，一轮完整训练约为：

```text
22400 / 8 = 2800 optimizer steps
```

建议在服务器使用 Qwen2.5-0.5B-Instruct、LoRA rank 8、max sequence length 2048，先训练一个完整 epoch，再在 1,000 条独立 V2.1 eval tasks 上评测。是否进入 GRPO，应看以下条件而不是只看总成功率：

- ranking 模板是否从 0% 变为稳定非零。
- correct-filter 后 candidate accuracy 是否明显高于 15%。
- 不同候选位置上的成功率是否接近。
- 工具格式和 invalid rate 是否保持稳定。

## 10. 面试表达

> V2 的 held-out 评测显示模型能正确选筛选控件，但筛选后的候选准确率只有 15%，并且不同模板会重复选择固定位置。我没有直接继续做 RL，而是先修复数据分布：生成 20 个训练页面实例和 5 个独立评测实例，对候选做循环换位，并为每个实例重新生成 element ID。最终 6,000 条训练任务中，同一 template-answer 至少出现在 4 个位置，单个模板的最大位置占比降到 26%；7,000 条 train/eval 任务全部通过 expert 验证。实现过程中还发现 20 个实例与 15 个模板存在 gcd 导致的上下文覆盖问题，因此改成按模板出现次数分配实例，并用测试固定这个不变量。
