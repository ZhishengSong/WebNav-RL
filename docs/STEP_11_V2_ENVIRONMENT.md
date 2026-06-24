# Step 11: V2 Structural-Generalization Environment

本步骤针对 V1 正式实验暴露的三个问题升级环境与数据：固定 element ID、页面结构单一、排序/筛选后的候选比较不足。

## 1. 为什么需要 V2

V1 observation 会显示实体信息，但不会显示对应的 clickable element ID。模型要成功点击，很容易依赖训练中反复出现的固定 ID 映射，例如 `shop_item_003`，而不是真正根据当前 observation 选择动作。

此外，V1 train/eval 使用相同页面结构，只随机切分任务文本。这能验证任务级泛化，但不能验证结构级网页导航能力。

V2 的核心问题变为：

> 当 eval 页面使用未见布局和全新的随机 element ID 时，模型能否读取 observation 中的 ID，并根据实体属性完成筛选、比较和点击？

## 2. 设计原则

### 随机但可见的 Element ID

V2 element ID 使用随机 token：

```text
el_c19hmi03fh
el_809fsp8r28
```

每个可点击项在 observation 中显式显示：

```text
[element_id=el_c19hmi03fh] Filter price under 100
CARD(el_809fsp8r28) name=FocusLamp Max, rating=4.9, cost=$78
```

随机 ID 防止语义猜测，显式显示保证任务可解。模型必须从当前 observation 复制正确 ID。

### 结构级 Train/Eval Split

| Layout | Split | Style |
| --- | --- | --- |
| `train_a` | train | sentence/list |
| `train_b` | train | compact/table-like |
| `eval_c` | eval | grid/record-like |

三套布局分别生成独立 ID。审计结果：

```text
train_eval_element_id_overlap = 0
expert_element_id_overlap = 0
```

### 干扰项

首页加入 help、delivery、enrollment 和 map 等可点击入口。它们是合法 element，但不会提供商品或课程候选。这样模型不仅要生成合法 ID，还要识别与目标相关的控件。

## 3. 数据规模

页面：

- 24 个商品，6 个 category。
- 24 门课程，6 个 department。
- 每个布局 70 页。
- 总计 210 页。
- 每个布局 179 个随机 element ID。

任务：

- 3000 train tasks。
- 500 held-out eval tasks。
- 15 个任务模板。
- train 每个模板严格 200 条。
- eval 每个模板 33 或 34 条。
- easy 932、medium 466、hard 2102。

训练数据：

- 3000 train expert trajectories。
- 500 eval expert trajectories。
- 11200 train next-action examples。
- 1868 eval next-action examples。

## 4. Targeted Task Templates

Shopping：

- name lookup。
- price lookup。
- color + category after filtering。
- category highest rating。
- category lowest price。
- category + budget + highest rating。
- under $100 highest rating。
- under $100 lowest price。

Course：

- code lookup。
- title lookup。
- department + time after filtering。
- department highest rating。
- department + credits + highest rating。
- credits + department。
- credits highest rating。

这些模板直接覆盖 V1 中没有稳定改善的 candidate-after-filter 和多条件比较。

## 5. Expert Verification

执行命令：

```bash
python scripts/run_v2_data.py --train-tasks 3000 --eval-tasks 500 --seed 31
```

结果：

```json
{
  "num_tasks": 3500,
  "expert_successes": 3500,
  "expert_success_rate": 1.0,
  "invalid_actions": 0,
  "action_errors": 0,
  "expert_path_target_matches": 3500,
  "average_steps": 3.7337,
  "train_next_action_examples": 11200,
  "eval_next_action_examples": 1868
}
```

额外使用通用 eval pipeline 在 held-out layout C 上回放 20 条任务：

```text
success: 20/20
format accuracy: 100%
invalid tool calls: 0
```

## 6. 生成产物

```text
pages/generated_pages_v2/metadata.json
pages/generated_pages_v2/manifest.json
tasks/v2/train_tasks.jsonl
tasks/v2/eval_tasks.jsonl
tasks/v2/manifest.json
outputs/trajectories/v2_expert_train_trajectories.jsonl
outputs/trajectories/v2_expert_eval_trajectories.jsonl
training/v2/sft_train.jsonl
training/v2/sft_eval.jsonl
outputs/eval_reports/v2_data_report.json
```

大体积生成文件默认被 `.gitignore` 排除，可以通过脚本确定性重建。小型审计报告保留在仓库中。

## 7. 通用接口升级

Expert runner、model eval 和 GRPO rollout 都支持显式 metadata：

```bash
python scripts/run_eval.py \
  --tasks tasks/v2/eval_tasks.jsonl \
  --metadata pages/generated_pages_v2/metadata.json
```

GRPO rollout 同样可以传：

```bash
python training/grpo_rollout.py \
  --tasks tasks/v2/train_tasks.jsonl \
  --metadata pages/generated_pages_v2/metadata.json
```

默认不传 `--metadata` 时仍使用 V1 页面，保持向后兼容。

## 8. 测试覆盖

`tests/test_v2_data.py` 验证：

- train/eval layout 完全分离。
- train/eval element ID 零重叠。
- ID 满足随机 token 格式，不包含实体语义。
- 15 个模板采样均衡。
- held-out instruction 明示 start page。
- 每个 expert click ID 在点击前出现在 observation。
- expert 路径最终 detail page 的 answer 与 task target 一致。
- expert 和通用 model runner 都能完成 V2 task。

## 9. 面试时怎么讲

> 第一版环境虽然有 held-out task，但页面和 element ID 固定，模型可能记住 ID 映射。第二版我把 element ID 随机化并显式放进 observation，用 train A/B 和未见的 eval C 做结构级切分。任务也从简单实体查找升级为均衡的筛选、排序和多条件候选比较。3500 条 expert 轨迹全部验证成功，下一步的模型指标会更接近真正的 observation-grounded navigation，而不是固定 ID 记忆。

## 10. 当前边界与下一步

V2 已完成环境和数据 readiness，但尚未产生模型结果，因此不能声称模型已经实现结构泛化。

下一步是 Step 12：

1. 用 11200 个 next-action examples 训练 V2 LoRA SFT。
2. 在 layout C 的 500 tasks 上评测。
3. 按 easy/medium/hard 和 15 个模板分析。
4. 与 V1 的固定布局结果分开报告，不直接比较绝对成功率。
