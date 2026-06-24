# WebNav-RL 项目 Todo：面向本地网页导航的 Agentic RL 后训练系统

## 0. 项目定位

项目目标：构建一个低成本、可验证、可扩展的 Mini Browser Agentic RL Post-training 项目。

核心思路：

> 构建一个本地可控网页环境，让小模型通过工具调用完成网页搜索、点击、信息提取和答案提交任务；先用 expert trajectory 做 SFT，再用 rule-based verifier 产生 reward，通过 GRPO 做 Agentic RL 后训练，并评估 Base / SFT / GRPO 的提升。

项目重点不是做真实浏览器自动化，而是做一个 **可验证的 agentic RL 训练环境**。

---

## 1. 项目总流程

完整流程如下：

```text
本地网页生成
→ 任务生成
→ 工具环境构建
→ Expert trajectory 生成
→ SFT 数据构造
→ LoRA SFT
→ Agent rollout
→ Reward / Verifier
→ GRPO 后训练
→ Base / SFT / GRPO 评估
→ Error Analysis / README / Demo
```

## 1.1 当前进度快照（2026-06-24）

当前仓库已经完成：

```text
V0 本地环境与工具闭环
V1 1000-task 数据、Qwen2.5-0.5B LoRA SFT 与完整评测
Reward / Error Analysis / Group Rollout
Reference-policy KL 的 GRPO-style LoRA 更新
3 个训练 seed 的 200-task 配对评测
V2 随机 ID、多布局、结构级 held-out 数据环境
```

V1 受控服务器实验：

| Run | Success rate | Delta vs SFT |
| --- | ---: | ---: |
| SFT baseline | 60.5% | - |
| GRPO-KL seed 7 | 64.0% | +3.5 pp |
| GRPO-KL seed 17 | 63.0% | +2.5 pp |
| GRPO-KL seed 29 | 59.5% | -1.0 pp |

多 seed 汇总：

```text
GRPO-KL mean success rate: 62.17%
sample standard deviation: 2.36 pp
mean delta vs SFT: +1.67 pp
best-seed McNemar p-value: 0.1435
```

严谨结论：GRPO-KL 存在初步正向信号，但 seed 方差明显，尚未证明稳定显著提升。

V2 结构泛化数据：

```json
{
  "pages": 210,
  "train_tasks": 3000,
  "eval_tasks": 500,
  "templates": 15,
  "train_next_action_examples": 11200,
  "expert_successes": 3500,
  "expert_path_target_matches": 3500,
  "invalid_actions": 0,
  "train_eval_element_id_overlap": 0
}
```

V2 使用 train A/B 两种布局和 held-out eval C，所有 element ID 都是随机 token，并在 observation 中显式展示。15 个模板在 train 中各 200 条，重点覆盖排序、筛选和多条件候选比较。

**当前正在进行 Step 12：V2 SFT baseline。**

```text
model: Qwen2.5-0.5B-Instruct
train data: training/v2/sft_train.jsonl
next-action examples: 11200
planned optimizer steps: 1400
held-out eval: layout C / 500 tasks
```

当前优先级：

```text
1. 完成 V2 LoRA SFT step1400
2. 在 held-out layout C 上跑 500-task eval
3. 按 15 个模板和 difficulty 做错误分析
4. 根据 V2 baseline 决定是否进入 V2 GRPO
```

---

## 2. 推荐目录结构

```text
WebNav-RL/
├── env/
│   ├── browser_env.py
│   ├── page_state.py
│   └── verifier.py
│
├── pages/
│   ├── html_templates/
│   ├── generated_pages/
│   └── page_generator.py
│
├── tasks/
│   ├── task_generator.py
│   ├── task_loader.py
│   ├── train_tasks.jsonl
│   └── eval_tasks.jsonl
│
├── tools/
│   ├── tool_registry.py
│   └── web_tools.py
│
├── rollout/
│   ├── rollout_runner.py
│   ├── trajectory.py
│   └── parser.py
│
├── training/
│   ├── build_sft_data.py
│   ├── sft_train.py
│   ├── grpo_train.py
│   └── reward.py
│
├── eval/
│   ├── evaluate.py
│   ├── metrics.py
│   └── error_analysis.py
│
├── scripts/
│   ├── generate_pages.sh
│   ├── generate_tasks.sh
│   ├── run_sft.sh
│   ├── run_grpo.sh
│   └── run_eval.sh
│
├── outputs/
│   ├── trajectories/
│   ├── checkpoints/
│   ├── eval_reports/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

# Version 0：环境 Demo 版

## 目标

不训练模型，先把本地网页环境、工具接口、任务系统和 verifier 跑通。

完成后，你应该有一个可以本地运行的 mini browser environment。

---

## V0 Todo

### 0.1 项目初始化

- [x] 创建 GitHub repo：`WebNav-RL`
- [x] 创建基础目录结构
- [x] 创建 `requirements.txt`
- [x] 创建初版 `README.md`
- [x] 写清楚项目目标和整体流程

---

### 0.2 构造本地网页环境

第一版只做 2 类网页：

```text
电商商品搜索页面
学校课程查询页面
```

每类网页先生成 20–50 个页面。

网页元素建议包括：

```text
搜索框
筛选按钮
商品/课程列表
详情页
提交答案按钮
```

Todo：

- [x] 写电商页面 HTML 模板
- [x] 写课程页面 HTML 模板
- [x] 写页面数据生成脚本
- [x] 给每个可点击元素分配 `element_id`
- [x] 保存页面状态 metadata
- [x] 生成静态 HTML 页面

当前实现：`pages/page_generator.py` 生成 shopping/course 两类页面，共 26 个本地 HTML 页面，并保存 `pages/generated_pages/metadata.json`。

示例页面 metadata：

```json
{
  "page_id": "shop_home_001",
  "page_type": "shopping",
  "elements": [
    {"element_id": "item_001", "text": "SoundCore A20", "target_page": "item_detail_001"},
    {"element_id": "filter_price_under_100", "text": "Price < 100", "target_page": "shop_filtered_001"}
  ]
}
```

---

### 0.3 定义工具接口

第一版只保留 4 个工具：

```python
open_page(page_id)
click(element_id)
get_visible_text()
submit_answer(answer)
```

Todo：

- [x] 实现 `open_page(page_id)`
- [x] 实现 `click(element_id)`
- [x] 实现 `get_visible_text()`
- [x] 实现 `submit_answer(answer)`
- [x] 维护当前页面状态
- [x] 记录每一步 action
- [x] 返回标准化 tool response

当前实现：`env/browser_env.py` 已维护 episode state、action log、invalid action count 和标准化 tool response。

工具返回格式示例：

```json
{
  "tool_name": "click",
  "status": "success",
  "current_page": "item_detail_001",
  "observation": "Opened detail page for SoundCore A20"
}
```

---

### 0.4 构造任务

任务格式示例：

```json
{
  "task_id": "shop_001",
  "start_page": "shop_home_001",
  "instruction": "找到价格低于100美元且评分最高的蓝牙耳机。",
  "target_answer": "SoundCore A20",
  "difficulty": "easy",
  "page_type": "shopping"
}
```

Todo：

- [x] 构造 20 条 V0 demo 任务
- [x] 实现 task loader
- [x] 支持 `env.reset(task)`
- [x] 支持最大步数限制字段，例如 8 步

当前实现：`scripts/run_v0_demo.py` 可生成并运行 20 条 demo 任务；`tasks/task_loader.py` 可加载 JSONL 任务文件。

---

### 0.5 实现 verifier

第一版 verifier 很简单：

```text
submit_answer == target_answer → success
否则 failure
```

Todo：

- [x] 实现 exact match verifier
- [x] 记录 success / failure
- [x] 记录 step count
- [x] 记录 invalid action
- [x] 输出 episode summary

Episode summary 示例：

```json
{
  "task_id": "shop_001",
  "success": true,
  "final_answer": "SoundCore A20",
  "target_answer": "SoundCore A20",
  "steps": 4,
  "invalid_actions": 0
}
```

---

### 0.6 写 rule-based expert

先不用模型，让专家策略能完成任务。

Todo：

- [x] 根据 task metadata 写 hard-coded expert
- [x] expert 调用 `open_page / click / submit_answer`
- [x] 保存完整 trajectory
- [x] 在 20 条任务上测试 success rate

完成标准：

```text
rule-based expert 在 20 条任务上 success rate = 100%
所有工具调用日志可以保存
```

当前验证结果：V0 demo 成功率 20/20，`success_rate = 1.0`，`invalid_actions = 0`，`avg_steps = 3.25`。

---

# Version 1：SFT 跑通版

## 目标

让小模型学会网页工具调用格式。

完成后，你可以展示：

```text
Base model 工具调用不稳定
SFT 后工具调用格式明显更稳定
```

---

## V1 Todo

### 1.1 扩大任务集

从 20 条任务扩到 300–1000 条。

任务类型先控制在 3 种：

```text
找最高评分商品
找最低价格商品
找满足条件的课程/商品
```

Todo：

- [x] 写 shopping task generator
- [x] 写 course task generator
- [x] 自动生成商品/课程数据
- [x] 自动计算 target answer
- [x] 自动保存 task jsonl
- [x] 划分 train/eval

当前实现：`tasks/task_generator.py` 可生成 1000 条任务，并按 800/200 划分 `train_tasks.jsonl` 和 `eval_tasks.jsonl`。

建议数据量：

```text
train tasks: 800
 eval tasks: 200
```

---

### 1.2 生成 expert trajectories

把 rule-based expert 跑在所有任务上，生成 SFT 数据。

Trajectory 格式示例：

```json
{
  "instruction": "找到价格低于100美元且评分最高的蓝牙耳机。",
  "messages": [
    {"role": "user", "content": "找到价格低于100美元且评分最高的蓝牙耳机。"},
    {"role": "assistant", "content": "<tool_call>{\"name\": \"open_page\", \"arguments\": {\"page_id\": \"shop_home_001\"}}</tool_call>"},
    {"role": "tool", "content": "Opened shop_home_001"},
    {"role": "assistant", "content": "<tool_call>{\"name\": \"click\", \"arguments\": {\"element_id\": \"filter_price_under_100\"}}</tool_call>"},
    {"role": "tool", "content": "Filtered items under 100 dollars"},
    {"role": "assistant", "content": "<tool_call>{\"name\": \"submit_answer\", \"arguments\": {\"answer\": \"SoundCore A20\"}}</tool_call>"}
  ]
}
```

Todo：

- [x] 保存 expert trajectory
- [x] 转换成 chat format
- [x] 转换成模型训练格式
- [x] 划分 train/eval
- [x] 检查 tool call JSON 合法性

当前实现：`rollout/rollout_runner.py` 生成 expert trajectories，`training/build_sft_data.py` 转换并校验 tool call JSON；当前已有 800 条 SFT train example 和 200 条 SFT eval example。

---

### 1.3 选择模型

第一版建议：

```text
Qwen2.5-0.5B-Instruct
或者 Qwen3-0.6B
```

Todo：

- [x] 加载 base model
- [x] 加载 tokenizer
- [x] 设置 chat template
- [x] 写 LoRA 配置
- [x] 写 SFT 训练脚本

当前实现：已接入 Qwen2.5-0.5B-Instruct，并完成 V1 step200 SFT、服务器复现和 V2 step1400 训练准备。

建议只做 LoRA，不做全参训练。

---

### 1.4 SFT 训练

训练目标：

```text
模型根据 instruction 和 tool response，生成正确 tool call
```

Todo：

- [x] 跑 V1 LoRA SFT
- [x] 保存 checkpoint
- [x] 记录 loss 日志
- [x] 保存训练 metadata
- [ ] 完成 V2 LoRA SFT step1400（当前进行中）

建议配置：

```text
model: Qwen2.5-0.5B-Instruct / Qwen3-0.6B
method: LoRA SFT
max_seq_len: 2048
batch_size: 小 batch + gradient accumulation
epoch: 1–3
```

---

### 1.5 SFT Eval

评估指标：

```text
Tool Call Format Accuracy
Task Success Rate
Average Step Count
Invalid Tool Call Rate
```

Todo：

- [x] 写 rollout runner
- [x] 让 base model 跑 eval tasks
- [x] 让 SFT model 跑 eval tasks
- [x] 比较指标
- [x] 输出 `eval_report.json`
- [x] 保存失败案例
- [ ] 完成 V2 held-out layout C 的 500-task SFT eval

当前状态：V1 Base/SFT/GRPO 已完成完整对比。V2 eval runner 已支持 `--metadata`，并通过 held-out layout expert replay smoke；待 V2 SFT checkpoint 完成后运行正式 500-task eval。

完成标准：

```text
SFT model 的 tool call format accuracy 明显高于 base model
SFT model 有一定 task success rate
```

---

# Version 2：GRPO 最小 RL 版

## 目标

做出真正的 Agentic RL post-training 闭环。

完成后，这个项目已经可以正式写进简历。

---

## V2 Todo

### 2.1 设计 reward function

第一版 reward 不要太复杂：

```text
reward = 0
+ 0.2 tool call 格式正确
+ 0.3 页面路径正确
+ 0.4 最终答案正确
- 0.05 每次多余操作
- 0.2 invalid tool call
```

Todo：

- [x] 实现 format reward
- [x] 实现 path reward
- [x] 实现 final answer reward
- [x] 实现 step penalty
- [x] 实现 invalid action / no-submit penalty
- [x] 保存每条 trajectory 的 reward breakdown

Reward breakdown 示例：

```json
{
  "task_id": "shop_001",
  "format_reward": 0.2,
  "path_reward": 0.3,
  "answer_reward": 0.4,
  "step_penalty": -0.1,
  "invalid_penalty": 0.0,
  "total_reward": 0.8
}
```

---

### 2.2 改造 rollout runner

GRPO 需要对同一个 prompt 采样多个 response。

Todo：

- [x] 支持 `group_size = 4`
- [x] 同一个 task 采样多条 trajectory
- [x] 每条 trajectory 独立执行环境
- [x] 记录 reward
- [x] 记录 response
- [x] 记录 tool calls
- [x] 支持 incremental / resume rollout cache

建议第一版配置：

```text
RL tasks: 200–500
group size: 4
max steps: 6–8
max tokens: 512–1024
```

---

### 2.3 接入 GRPO 训练

第一版可以使用 TRL / Unsloth / 简化版 GRPO。

Todo：

- [x] 准备 grouped rollout dataset
- [x] 写 reward function wrapper
- [x] 接入模型生成
- [x] 计算 group relative advantage
- [x] 加入 frozen reference-policy KL
- [x] LoRA 更新
- [x] 保存 checkpoint
- [x] 记录 policy / KL / advantage 日志

训练目标不是 SOTA，而是证明：

```text
SFT + GRPO 比 SFT 在至少一个核心指标上提升。
```

---

### 2.4 GRPO Eval

对比模型：

```text
Base
SFT
SFT + GRPO
```

指标：

```text
Task Success Rate
Tool Call Format Accuracy
Invalid Tool Call Rate
Average Step Count
Final Answer Accuracy
```

Todo：

- [x] 固定独立 eval set
- [x] 跑 Base model
- [x] 跑 SFT model
- [x] 跑 SFT + GRPO model
- [x] 生成对比表格
- [ ] 生成柱状图
- [x] 保存失败案例
- [x] 运行 3 个 GRPO training seed
- [x] 做 McNemar 配对与模板级分析

完成标准：

```text
SFT + GRPO 比 SFT 有至少一个核心指标提升：
- success rate 提升
- invalid tool call rate 降低
- average step count 降低
```

---

# Version 2.5：V2 结构泛化数据版

## 目标

修复 V1 固定 element ID 和同布局随机切分的问题，验证 observation-grounded navigation。

### 2.5.1 页面与结构切分

- [x] 扩展到 24 个商品和 24 门课程
- [x] 生成 210 个页面
- [x] 实现 train A/B 两种布局
- [x] 实现 held-out eval C 布局
- [x] 使用随机无语义 element ID
- [x] 在 observation 中显式展示 clickable ID
- [x] 加入 help / map / delivery 等干扰入口
- [x] 验证 train/eval element ID overlap = 0

### 2.5.2 Targeted Tasks

- [x] 增加排序、筛选、多条件候选任务
- [x] 构造 15 个模板
- [x] train 每个模板严格 200 条
- [x] eval 每个模板 33–34 条
- [x] 生成 3000 train + 500 eval
- [x] 生成 11200 train next-action examples
- [x] 检查价格、属性和 ranking target 无歧义

### 2.5.3 数据验证

- [x] expert success 3500/3500
- [x] expert path target match 3500/3500
- [x] invalid actions = 0
- [x] 通用 eval runner 支持 `--metadata`
- [x] GRPO rollout 支持 `--metadata`
- [x] held-out layout C expert smoke 20/20
- [x] 写 `docs/STEP_11_V2_ENVIRONMENT.md`

### 2.5.4 V2 模型实验

- [ ] 完成 V2 SFT step1400（当前进行中）
- [ ] 在 held-out layout C 上评测 500 tasks
- [ ] 按 15 个模板统计成功率
- [ ] 按 easy / medium / hard 统计成功率
- [ ] 根据 baseline 决定是否启动 V2 GRPO

---

# Version 3：简历完整版本

## 目标

让项目经得起面试追问，而不是只有代码跑通。

---

## V3 Todo

### 3.1 增加任务难度分层

任务分成 4 个 level：

```text
Level 1：单页面信息查找
Level 2：筛选 + 排序
Level 3：多页面跳转
Level 4：需要对比多个候选项
```

Todo：

- [x] 给任务打 easy / medium / hard label
- [x] V2 生成 500 条均衡模板 eval tasks
- [ ] 完成 V2 模型后分别统计 difficulty 成功率
- [x] V1 已完成 template-level 提升/退化分析

---

### 3.2 增加网页类型

从 2 类扩到 4 类：

```text
电商商品页
学校课程页
租房信息页
论文搜索页
```

Todo：

- [ ] 新增租房页面生成器
- [ ] 新增论文页面生成器
- [ ] 新增对应 task templates
- [ ] 新增对应 expert strategy
- [ ] 扩展 eval set

---

### 3.3 Reward Ablation

对比：

```text
Full reward
No step penalty
No path reward
Only final answer reward
```

Todo：

- [ ] 跑小规模 GRPO ablation
- [ ] 比较 success rate
- [ ] 比较 average steps
- [ ] 比较 invalid tool call rate
- [ ] 写分析

---

### 3.4 Error Analysis

失败类型：

```text
JSON 格式错误
工具名错误
element_id 错误
页面跳转错误
答案提取错误
过早 submit
循环点击
```

Todo：

- [x] 写 error classifier
- [x] 统计错误占比
- [x] 保存代表性案例
- [x] 在 README / docs 中展示错误分析

---

### 3.5 项目文档

README 应包括：

```text
项目背景
为什么是 Agentic RL
环境设计
工具定义
任务生成
reward function
训练方法
实验结果
错误分析
未来改进
```

Todo：

- [x] 写完整 README
- [ ] 画系统架构图
- [ ] 画训练流程图
- [ ] 画指标对比图
- [ ] 录制 demo gif
- [x] 整理简历描述和面试问答
- [x] 写最终技术报告与 Step 01–11 文档

---

# Version 4：差异化 Scale 版

## 目标

让项目看起来不是 toy，而是一个可扩展框架。

---

## V4 Todo

### 4.1 Curriculum Task Generator

根据难度逐步生成任务：

```text
easy → medium → hard
```

Todo：

- [ ] 实现 difficulty controller
- [ ] 训练时先采 easy
- [ ] 成功率高后采 medium/hard
- [ ] 记录 curriculum 曲线

---

### 4.2 Failure-driven Data Mining

根据模型失败样本生成新任务。

Todo：

- [ ] 收集失败 trajectory
- [ ] 分类失败原因
- [ ] 针对失败原因生成相似任务
- [ ] 加入下一轮 SFT/RL 数据

示例：

```text
模型经常选错最高评分商品
→ 生成更多需要排序比较的任务
```

---

### 4.3 并行 Rollout

Todo：

- [ ] 支持 multiprocessing
- [ ] 支持 batch rollout
- [ ] 支持保存 rollout cache
- [ ] 支持断点续跑

---

### 4.4 通用 Env Interface

抽象出统一接口：

```python
class AgentEnv:
    def reset(self, task):
        pass

    def step(self, action):
        pass

    def verify(self):
        pass
```

Todo：

- [ ] 把 WebNav 环境抽象出来
- [ ] 抽象 tool registry
- [ ] 抽象 verifier
- [ ] 为未来接入 SQLEnv / CodeFixEnv 做准备

未来可扩展成：

```text
Verifiable Agentic RL Framework
├── WebNavEnv
├── SQLEnv
└── CodeFixEnv
```

---

# Version 5：论文 / 研究味版本

## 目标

让项目更像研究探索，而不只是工程项目。

---

## V5 Todo

### 5.1 对比不同 post-training 方法

对比：

```text
SFT
DPO
GRPO
GRPO + curriculum
```

Todo：

- [ ] 构造 preference pair
- [ ] 训练 DPO baseline
- [ ] 和 GRPO 对比
- [ ] 分析 DPO 和 GRPO 在 multi-step task 上的差异

---

### 5.2 分析 RL 对 agent 行为的影响

分析问题：

```text
GRPO 是否减少无效点击？
GRPO 是否减少过早 submit？
GRPO 是否提升 hard task 成功率？
GRPO 是否提升跨页面任务？
```

Todo：

- [ ] 统计 action distribution
- [ ] 统计 step count distribution
- [ ] 分析 reward hacking
- [ ] 分析模型行为变化

---

### 5.3 写技术报告

报告标题可以是：

```text
WebNav-RL: Verifiable Agentic Post-training for Small Language Models in Local Web Navigation
```

建议结构：

```text
Abstract
Introduction
Environment
Task Generation
Reward Design
Training
Experiments
Error Analysis
Limitations
```

Todo：

- [ ] 写技术报告初稿
- [ ] 加入实验表格
- [ ] 加入系统图
- [ ] 加入失败案例分析

---

# 推荐执行路线

## 第一阶段：Version 0 + Version 1

目标：一周内做出 SFT demo。

```text
本地环境跑通
expert trajectory 生成
LoRA SFT 跑通
Base vs SFT eval
```

---

## 第二阶段：Version 2

目标：两周内跑通 GRPO 闭环。

```text
reward function
multi-sample rollout
GRPO training
Base / SFT / GRPO 对比
```

---

## 第三阶段：Version 3

目标：整理成简历项目。

```text
任务难度分层
多网页类型
eval 表格
error analysis
README
demo gif
```

---

## 第四阶段：Version 4 / Version 5

目标：做差异化和研究味。

```text
curriculum
failure-driven data mining
parallel rollout
DPO vs GRPO
technical report
```

---

# 最小可行 Todo 总表

当前最小可行主线已经完成，正在推进 V2 结构泛化模型实验：

```text
[x] 创建 WebNav-RL repo
[x] 实现本地 HTML 页面生成器
[x] 实现 open_page / click / get_visible_text / submit_answer
[x] 构造 20 条 V0 demo 任务
[x] 实现 verifier
[x] 实现 rule-based expert
[x] 自动生成 1000 条任务
[x] 生成 expert trajectories
[x] 转成 SFT 数据
[x] 写模型 rollout parser
[x] 写 Base model eval runner
[x] 运行 Qwen Base model eval
[x] 用 Qwen2.5-0.5B 做 LoRA SFT
[x] 评估 Base vs SFT
[x] 实现 reward function
[x] 接入 GRPO-KL
[x] 评估 Base vs SFT vs GRPO
[x] 做多 seed、配对和错误分析
[x] 写 README、最终报告和面试文档
[x] 实现 V2 随机 ID 和结构级 held-out split
[x] 生成 3000/500 V2 tasks 与 11200 action examples
[ ] 完成 V2 SFT step1400
[ ] 完成 V2 held-out layout C 500-task eval
[ ] 画架构图和结果图
```

---

# 每个版本对应简历价值

| 版本 | 能否写简历 | 价值 |
|---|---|---|
| Version 0 | 不建议单独写 | 只是环境 demo |
| Version 1 | 可以弱写 | 有 SFT，但 RL 不够 |
| Version 2 | 可以正式写 | 有 Agentic RL 闭环 |
| Version 3 | 推荐写 | 完整项目 |
| Version 4 | 很加分 | 有 scale pipeline |
| Version 5 | 很强 | 有研究味和技术报告 |

---

# 建议最终完成度

现实目标：

```text
必须完成：Version 0 + Version 1 + Version 2
尽量完成：Version 3
有时间再做：Version 4
不用一开始追：Version 5
```

最终简历项目主线：

> 构建本地 WebNav 环境，通过 expert trajectory 做 SFT，再用 verifier reward 做 GRPO，最后评估 Base / SFT / GRPO 在网页导航任务上的差异。

这个流程完整、可控、成本低，而且不千篇一律。

---

# 简历描述草稿

**WebNav-RL：面向本地网页导航任务的小模型 Agentic RL 后训练系统**

- 构建本地可控 WebNav 环境，设计 `open_page`、`click`、`get_visible_text`、`submit_answer` 等工具接口，使小模型通过多轮工具调用完成商品检索、课程查询、租房信息筛选等网页导航任务。
- 基于规则专家生成 tool-use trajectories，并使用 Qwen 0.5B/0.6B 进行 LoRA SFT，使模型学习标准化工具调用格式和多步网页操作流程。
- 设计 rule-based verifier 与多维 reward function，将 tool call 格式、页面路径、最终答案正确性、无效操作和步数惩罚转化为奖励信号，并基于 GRPO 进行 Agentic RL 后训练。
- 对比 Base、SFT、SFT+GRPO 在 Task Success Rate、Tool Call Format Accuracy、Invalid Tool Call Rate、Average Step Count 等指标上的表现，并进行错误类型分析和 reward ablation。

