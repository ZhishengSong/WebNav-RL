# Step 09: GRPO Trainer With Reference KL

本步骤把之前的 minimal GRPO prototype 往正式 RL 训练方向推进了一步：在 advantage-weighted policy update 外，加入 frozen reference policy 和 KL penalty。

## 1. 为什么要做这一步

Step 08 的最小 GRPO 原型已经证明了数据流可以闭环：

- 能从 SFT policy 采样 grouped rollouts。
- 能用 reward function 给每条轨迹打分。
- 能在同一 task group 内计算 relative advantage。
- 能用 advantage 加权 sampled action loss 更新 LoRA adapter。
- 能保存新的 adapter 并重新评测。

但 Step 08 的 20 题评测从 SFT baseline 的约 70% 退化到 45%。这不是意外结果，主要原因是：

- rollout 数据太少；
- 没有 reference policy；
- 没有 KL 约束；
- negative-advantage 样本会把 sampled action 概率压低，但缺少“不要离原 SFT 太远”的稳定器；
- 当前 reward 和采样规模还不足以支撑大幅 policy shift。

所以 Step 09 的目标不是立刻追求更高成功率，而是先补上 RL post-training 里非常关键的稳定项：reference-policy KL。

## 2. 本次代码改动

主要修改文件：

- `training/grpo_train.py`
- `tests/test_grpo_train.py`

新增能力：

- `--reference-adapter`：指定 frozen reference adapter，默认等于 `--adapter`。
- `--kl-beta`：KL penalty 系数，默认 `0.02`；设为 `0` 可关闭 reference model。
- `build_reference_model()`：加载不可训练的 reference policy。
- `token_log_probs()`：抽取 target action token 的 log probability。
- `sequence_kl_penalty()`：计算 sampled target token 上的 KL 近似。
- metadata 记录 `reference_adapter`、`kl_beta`、`kl_estimator` 和每步 KL 日志。

训练 loss 现在变为：

```text
loss = advantage_weighted_policy_loss + kl_beta * kl_loss
```

其中 policy loss 仍然来自 sampled assistant action：

```text
positive advantage -> 降低该 action 的 CE，提高概率
negative advantage -> 提高该 action 的 CE，降低概率
```

KL 部分使用 sampled token 上的 k3 estimator：

```text
log_ratio = ref_logp - policy_logp
kl = exp(log_ratio) - log_ratio - 1
```

这个估计量的好处是非负、数值更稳，适合在 sampled action token 上做轻量约束。

## 3. 为什么 reference 默认用 SFT adapter

当前最可靠的 policy 是：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200
```

它已经在 200 道 held-out 任务上达到：

- success rate: 63.5%
- tool-call format accuracy: 100%
- invalid tool-call rate: 1.88%

因此 GRPO 阶段不应该一开始就远离它。reference policy 使用同一个 SFT adapter，含义是：

> 允许 policy 根据 reward/advantage 做局部调整，但用 KL 惩罚限制它不要快速破坏 SFT 已经学到的工具格式和基本导航能力。

## 4. Smoke Run

本次做了一个 1-step smoke run，确认双模型前向、KL 计算、LoRA 更新、adapter 保存和 metadata 记录都能闭环。

命令：

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe training\grpo_train.py --model models\qwen2.5-0.5b-instruct --adapter outputs\checkpoints\qwen2_5_0_5b_lora_sft_step200 --rollouts outputs\rollouts\grpo_sft_step200_group4_task4.jsonl --output-dir outputs\checkpoints\qwen2_5_0_5b_lora_grpo_kl_smoke_step1 --max-steps 1 --batch-size 1 --gradient-accumulation-steps 1 --learning-rate 1e-5 --kl-beta 0.02 --log-every 1
```

输出 adapter：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_grpo_kl_smoke_step1
```

关键日志：

```json
{
  "optimizer_step": 1,
  "loss": 0.0001265406608581543,
  "policy_loss": 0.0001265406608581543,
  "kl_loss": 0.0,
  "kl_beta": 0.02,
  "mean_advantage": 0.13750000298023224,
  "mean_policy_log_prob": -0.0009202957153320312,
  "mean_reference_log_prob": -0.0009093284606933594
}
```

这里 `kl_loss` 初始为 `0.0` 是合理的，因为 policy 和 reference 在训练开始时加载的是同一个 SFT adapter。KL 的价值会在多个 optimizer steps 后体现：policy 被 reward 推动偏移时，KL 会约束它不要偏移太快。

metadata 文件：

```text
outputs/checkpoints/qwen2_5_0_5b_lora_grpo_kl_smoke_step1/grpo_training_metadata.json
```

## 5. 验证结果

普通基础测试：

```text
python -m pytest -q
8 passed, 2 skipped
```

这里 2 个 skipped 是 GRPO 数值测试，因为默认 Python 环境没有安装 torch。模型训练依赖本来就是 optional dependency。

模型环境中的 GRPO 数值测试：

```powershell
$env:PYTHONPATH='D:\Program\Anaconda\envs\research\Lib\site-packages;D:\job\Program\WebNav-RL\.python_deps\research'
D:\Program\Anaconda\envs\research\python.exe -m unittest tests.test_grpo_train -v
```

结果：

```text
Ran 2 tests
OK
```

编译检查：

```text
python -m compileall training eval rollout scripts tests
passed
```

## 6. 面试时怎么讲

可以这样解释本步骤：

> 我在最小 GRPO 原型上补了 reference-policy KL。之前的原型只根据 group-relative advantage 去增减 sampled action 的概率，容易在小数据上把 SFT policy 拉偏。现在训练时会同时加载一个 frozen SFT reference policy，对 sampled action token 计算 policy 和 reference 的 log probability，并加入 KL penalty。这样 RL 更新既能利用 reward 信号，又不会快速破坏 SFT 已经学到的工具调用格式和基础导航能力。

如果面试官问为什么 smoke run 的 KL 是 0：

> 因为 policy 和 reference 初始都来自同一个 SFT adapter，所以第一步更新前两者分布几乎相同。KL 约束的意义不是让初始 KL 很大，而是在后续多步训练中限制 policy drift。

如果面试官问这一步有没有提升指标：

> 这一步主要是算法稳定性建设，还不是最终性能实验。我做的是 1-step smoke run，验证双模型 forward、KL loss、adapter 保存和 metadata 闭环。下一步要扩大 group rollout，再做 SFT vs GRPO-KL 的同 split 对比。

## 7. 下一步

建议下一步做一个本地中等规模实验：

1. 用 SFT adapter 采样更多 grouped rollouts，比如 20 tasks x group size 4。
2. 用 `training/grpo_train.py` 做 5 到 20 步 GRPO-KL 更新。
3. 在同一批 20 或 50 个 eval tasks 上比较：
   - SFT adapter
   - GRPO-KL adapter
4. 记录：
   - success rate
   - format accuracy
   - invalid tool-call rate
   - error type 分布
   - KL/loss 曲线

如果本地实验耗时明显变长，再考虑把 rollout 扩大到服务器或 Hugging Face。当前 Step 09 仍然可以在本机完成。
