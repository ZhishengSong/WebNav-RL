# WebNav-RL 已完成工作讲解

> 本文记录早期 V0/V1 数据阶段。项目已经继续完成 SFT、GRPO-KL 和多 seed
> 正式评测；最终结论请看 `docs/FINAL_PROJECT_REPORT.md`，面试表达请看
> `docs/INTERVIEW_QA.md`。

这份文档用来帮助你理解当前已经完成的 V0/V1 数据阶段，并准备面试时的项目讲述。

本文所描述的阶段还没有进入模型训练和 GRPO；当时完成的是一个可验证的本地 WebNav 环境、任务生成系统、规则专家轨迹生成，以及 SFT 数据构造流水线。它的价值在于先把 agentic post-training 最核心的“环境-工具-轨迹-验证”闭环搭稳。

## 1. 一句话介绍项目

WebNav-RL 是一个面向小模型 Agentic RL 后训练的本地网页导航环境。它构造了可控的 shopping/course 网页任务，让模型通过 `open_page`、`click`、`get_visible_text`、`submit_answer` 等工具完成多步导航和信息查找；当前阶段已经用 rule-based expert 自动生成了 1000 条可验证轨迹，并转换成 SFT chat 数据，为后续 LoRA SFT 和 GRPO 训练做准备。

面试中可以这样说：

> 我先没有直接上真实浏览器，而是做了一个本地可控的 mini WebNav environment。核心目标是让任务可验证、轨迹可复现、reward 可计算。现在已经完成环境、工具接口、任务生成、规则专家、轨迹保存和 SFT 数据构造，expert 在 1000 条任务上 100% 成功。

## 2. 为什么先做本地环境，而不直接用真实浏览器

真实网页自动化有很多干扰项，比如页面变化、网络不稳定、DOM 复杂、登录状态、广告、异步加载等。这个项目的重点不是展示 Selenium 或 Playwright 技巧，而是研究小模型如何学会多步工具调用和 agentic decision making。

所以当前设计选择是：

- 页面是本地生成的静态 HTML。
- 页面状态由 `metadata.json` 显式描述。
- 可点击元素有稳定的 `element_id`。
- 任务有明确 `target_answer`。
- verifier 可以 rule-based 判断 success/failure。

这样的好处：

- 训练数据可以大规模自动生成。
- expert trajectory 可以稳定复现。
- 后续 RL reward 不依赖人工标注。
- Base / SFT / GRPO 的评估可以在固定 eval set 上公平对比。

面试时如果被问“这是不是 toy”，可以回答：

> 它是刻意控制变量的 toy environment，但目标不是复刻真实网页，而是搭一个 verifiable agentic RL sandbox。真实浏览器环境复杂但难以稳定验证；我先把工具调用、轨迹、reward、评估这些核心训练闭环做成可控系统，再逐步扩展网页类型和任务难度。

## 3. 当前已经完成的整体流水线

当前一键入口是：

```bash
python scripts/run_v1_data.py --num-tasks 1000 --train-ratio 0.8 --seed 7
```

它会执行完整数据流水线：

```text
生成本地网页和 metadata
-> 自动生成 1000 条任务
-> 按 800/200 划分 train/eval
-> rule-based expert 执行工具调用
-> verifier 统计每条 episode 的结果
-> 保存 expert trajectories
-> 转换成 SFT chat 数据
```

当前验证结果：

```json
{
  "num_tasks": 1000,
  "train_tasks": 800,
  "eval_tasks": 200,
  "train_sft_examples": 800,
  "eval_sft_examples": 200,
  "success_rate": 1.0,
  "invalid_actions": 0,
  "avg_steps": 3.167
}
```

这说明目前 rule-based expert 在所有生成任务上都能完成，且没有非法点击或非法工具调用。

## 4. 目录和模块职责

当前核心文件如下：

```text
pages/page_generator.py
tasks/task_generator.py
tasks/task_loader.py
env/page_state.py
env/browser_env.py
env/verifier.py
tools/web_tools.py
tools/tool_registry.py
rollout/rollout_runner.py
rollout/trajectory.py
training/build_sft_data.py
scripts/run_v0_demo.py
scripts/run_v1_data.py
```

### 4.1 页面生成：`pages/page_generator.py`

这个模块负责生成本地页面和页面 metadata。

目前支持两类页面：

- shopping：商品搜索/筛选/详情页。
- course：课程查询/筛选/详情页。

页面数据来自两个 dataclass：

```python
Product(name, category, price, rating, color)
Course(code, title, department, credits, rating, time)
```

生成出来的页面包括：

- `shop_home.html`
- `shop_under_100.html`
- `shop_rating_desc.html`
- `shop_item_001.html` 到 `shop_item_010.html`
- `course_home.html`
- `course_cs.html`
- `course_credits_4.html`
- `course_detail_001.html` 到 `course_detail_010.html`

同时生成：

```text
pages/generated_pages/metadata.json
```

metadata 是环境真正使用的状态来源。它会记录：

```json
{
  "page_id": "shop_home",
  "page_type": "shopping",
  "visible_text": "...",
  "elements": [
    {
      "element_id": "filter_price_under_100",
      "text": "Filter price under 100",
      "target_page": "shop_under_100"
    }
  ],
  "html_path": "pages/generated_pages/shop_home.html"
}
```

面试讲法：

> 页面 HTML 是给人看的，metadata 是给环境执行用的。我把可见文本、可点击元素和跳转目标都结构化保存，这样 agent 执行 click 时不需要真实浏览器，也能得到确定的 observation。

### 4.2 任务生成：`tasks/task_generator.py`

这个模块负责自动构造任务。

任务类型目前包括：

- 按名称查找商品或课程。
- 按颜色/类别查找商品。
- 按价格查找商品。
- 找最高评分商品。
- 找最低价格商品。
- 找价格低于 100 美元的最高评分/最低价格商品。
- 按 course code/title 查找课程。
- 按 department/time 查找课程。
- 找最高评分课程。
- 找 4 credit 课程。

每条任务包含：

```json
{
  "task_id": "shop_00001",
  "start_page": "shop_home",
  "instruction": "Please find the lowest priced item under 100 dollars. Use the page information.",
  "target_answer": "TravelMug One",
  "difficulty": "medium",
  "page_type": "shopping",
  "template": "shopping_under_100_lowest_price",
  "expert_clicks": ["filter_price_under_100", "shop_under_100_item_007"],
  "max_steps": 8,
  "split": "train"
}
```

这里最关键的是 `expert_clicks`。它是 rule-based expert 的最短路径，用来生成专家轨迹。

设计上有两个层次：

- task instruction 是给模型看的自然语言。
- expert_clicks 是给专家策略使用的 oracle path。

面试讲法：

> 任务生成时我不只生成 instruction 和 answer，还同时生成 expert path。这样后续可以自动构造 SFT 数据，不需要人工演示每条网页操作。

### 4.3 页面状态：`env/page_state.py`

`PageStore` 负责读取 `metadata.json`，并提供两个核心能力：

- `get(page_id)`：根据 page id 找页面状态。
- `find_element(page_id, element_id)`：在当前页找可点击元素。

这个模块把“页面数据库”和“环境执行逻辑”分开。后面如果要换成更复杂的页面来源，比如 SQLite、真实 DOM snapshot、Playwright observation，只需要替换 PageStore 或扩展它。

### 4.4 环境工具：`env/browser_env.py`

这是当前最核心的环境模块。它维护 episode 状态：

```python
self.task
self.current_page
self.final_answer
self.done
self.invalid_actions
self.action_log
```

实现了四个工具：

```python
open_page(page_id)
click(element_id)
get_visible_text()
submit_answer(answer)
```

每个工具都会返回标准化 response：

```json
{
  "tool_name": "click",
  "arguments": {"element_id": "filter_price_under_100"},
  "status": "success",
  "current_page": "shop_under_100",
  "observation": "Products with price under 100. ..."
}
```

如果模型点击了不存在的 element，会返回 error，并增加 `invalid_actions`：

```json
{
  "tool_name": "click",
  "status": "error",
  "observation": "Element not found: ..."
}
```

面试讲法：

> BrowserEnv 其实是一个轻量级 MDP 环境。state 是 current_page 和 action history，action 是工具调用，observation 是页面 visible_text，terminal action 是 submit_answer。这样可以自然接 rollout runner 和 RL reward。

### 4.5 Verifier：`env/verifier.py`

当前 verifier 是 exact match：

```python
answer.strip().lower() == target_answer.strip().lower()
```

每个 episode 输出 summary：

```json
{
  "task_id": "shop_00001",
  "success": true,
  "final_answer": "TravelMug One",
  "target_answer": "TravelMug One",
  "steps": 4,
  "invalid_actions": 0
}
```

当前 verifier 很简单，但这是故意的。第一阶段先保证评价信号稳定，后续可以扩展成：

- exact match + alias match。
- 数值容忍。
- 多答案集合。
- path correctness。
- reward breakdown。

面试讲法：

> 我先用 exact match 作为最小 verifier，因为任务答案都是规范化的商品名或课程 code。后续 GRPO 阶段会把 verifier 扩展成多维 reward：格式正确、路径正确、最终答案正确、invalid action penalty、step penalty。

### 4.6 Tool Registry：`tools/tool_registry.py`

`ToolRegistry` 把工具名映射到具体函数：

```python
open_page -> tools.open_page
click -> tools.click
get_visible_text -> tools.get_visible_text
submit_answer -> tools.submit_answer
```

目前 rule-based expert 直接调用 env 方法，registry 先作为接口预留。等接模型 rollout 时，模型会输出：

```json
{"name": "click", "arguments": {"element_id": "xxx"}}
```

parser 解析后就可以通过 ToolRegistry dispatch 到具体工具。

面试讲法：

> 我把工具注册表单独抽出来，是为了后续模型生成 tool call 后可以统一解析和 dispatch，也方便统计 unknown tool、参数错误等 invalid tool call。

### 4.7 Rule-based Expert：`rollout/rollout_runner.py`

rule-based expert 的逻辑很直接：

1. `env.reset(task)` 打开起始页面。
2. 根据 `task["expert_clicks"]` 依次 click。
3. 调用 `submit_answer(target_answer)`。
4. 保存完整 messages、actions 和 summary。

生成的轨迹格式：

```json
{
  "task_id": "shop_00001",
  "instruction": "...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<tool_call>{...}</tool_call>"},
    {"role": "tool", "content": "Shopping home. ..."}
  ],
  "actions": [...],
  "summary": {...}
}
```

这一步的目的不是让 expert 很智能，而是稳定地产生“正确示范”。后续 SFT 学的是：

- 在某个 instruction 下先调用什么工具。
- 看到 tool observation 后下一步怎么调用。
- 最后什么时候 submit_answer。
- tool call 的 JSON 格式。

面试讲法：

> 这里的 expert 是 oracle policy，它知道任务生成时计算出的 expert_clicks。它的职责不是解决未知任务，而是自动产出高质量 tool-use demonstrations，用于 SFT。

### 4.8 SFT 数据构造：`training/build_sft_data.py`

这个模块把 expert trajectories 转成训练数据。

它会做一件重要的校验：所有 assistant 消息必须符合：

```text
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
```

并检查：

- assistant 内容必须是 tool call。
- tool call 中必须有 `name` 和 `arguments`。
- `arguments` 必须是 JSON object。

最终生成：

```text
training/sft_train.jsonl
training/sft_eval.jsonl
```

每行格式：

```json
{
  "id": "shop_00001",
  "instruction": "...",
  "messages": [...],
  "summary": {...}
}
```

面试讲法：

> 在构造 SFT 数据时我加了 tool-call JSON 校验，避免把格式错误的 demonstration 写进训练集。因为这个项目里 tool-call format accuracy 是核心指标之一，数据质量会直接影响 SFT 后模型是否能稳定调用工具。

## 5. 当前产物

运行 V1 后，当前产物包括：

```text
pages/generated_pages/*.html
pages/generated_pages/metadata.json
tasks/all_tasks.jsonl
tasks/train_tasks.jsonl
tasks/eval_tasks.jsonl
outputs/trajectories/expert_train_trajectories.jsonl
outputs/trajectories/expert_eval_trajectories.jsonl
training/sft_train.jsonl
training/sft_eval.jsonl
```

数量：

```text
pages: 26
tasks: 1000
train tasks: 800
eval tasks: 200
SFT train examples: 800
SFT eval examples: 200
expert success rate: 100%
invalid actions: 0
```

## 6. 当前阶段的核心技术点

### 6.1 可验证环境

每个任务都有明确 target answer，环境中每一步工具调用都有确定 response，最终答案由 verifier 自动判断。

可以讲：

> 我把网页导航任务转化为可验证的 episodic environment。相比开放式网页问答，这种设计能稳定计算 success rate、invalid action rate、average step count，也能作为后续 RL reward 的基础。

### 6.2 工具调用格式标准化

assistant 的动作统一表示为：

```text
<tool_call>{"name": "click", "arguments": {"element_id": "..."}}</tool_call>
```

这让后续可以清晰区分：

- 模型是否输出了合法 JSON。
- 工具名是否存在。
- 参数是否正确。
- 环境执行是否成功。

可以讲：

> SFT 的第一目标不是让模型马上学会复杂推理，而是让小模型稳定学会 tool call protocol。只有格式稳定了，后续 RL rollout 才不会大量浪费在 parser error 上。

### 6.3 Expert trajectory 自动生成

任务生成器同时产出 `expert_clicks`，所以可以批量生成 demonstrations。

可以讲：

> 这相当于用程序化数据合成替代人工标注。我先用小规模实体库和模板任务保证路径可计算，再通过 paraphrase 增加 instruction 变化。

### 6.4 Train/eval 固定划分

当前按 0.8/0.2 划分：

```text
800 train
200 eval
```

后续模型训练时可以固定 eval set，对比：

```text
Base
SFT
SFT + GRPO
```

指标包括：

- Task Success Rate。
- Tool Call Format Accuracy。
- Invalid Tool Call Rate。
- Average Step Count。
- Final Answer Accuracy。

## 7. 面试讲述顺序

建议按这个顺序讲，不容易散：

1. 我想做的是小模型 agentic post-training，不是普通网页爬虫。
2. 所以先搭了一个本地可验证 WebNav 环境。
3. 环境有 shopping/course 两类网页，每个可点击元素都有 element_id。
4. agent 只能通过四个工具和环境交互。
5. 每条任务有 instruction、start page、target answer、expert path。
6. rule-based expert 自动生成 tool-use trajectory。
7. verifier 自动判断 success，并记录 steps/invalid actions。
8. 目前生成了 1000 条任务和 1000 条 expert trajectories，转成了 800/200 的 SFT 数据。
9. 下一步是 LoRA SFT，让小模型学会工具调用格式，再接 GRPO 用 verifier reward 做后训练。

一段完整口述可以这样说：

> 我这个项目的目标是做一个小模型 Agentic RL 后训练系统。因为真实网页环境不稳定、难验证，所以我先构建了一个本地可控的 WebNav sandbox。里面有商品搜索和课程查询两类页面，页面状态用 metadata 表示，每个按钮和条目都有稳定的 element_id。Agent 只能调用 open_page、click、get_visible_text、submit_answer 四个工具完成任务。任务生成器会自动生成 instruction、target answer 和 expert path；然后 rule-based expert 根据 expert path 生成完整 tool-use trajectory。每条 trajectory 都会经过 exact-match verifier，记录 success、step count 和 invalid actions。现在已经生成了 1000 条任务，800 条 train、200 条 eval，expert 成功率 100%，并且已经转成 SFT chat 数据。后续我会用这些轨迹做 LoRA SFT，再用 verifier reward 做 GRPO，比较 Base/SFT/GRPO 的 success rate 和 invalid action rate。

## 8. 常见追问和回答

### Q1：为什么不用真实浏览器？

答：

> 真实浏览器更接近应用，但它引入很多非训练核心的问题，比如网络、页面异步、DOM 变化、广告和登录。这个项目现阶段重点是 agentic post-training，所以我先做本地可验证环境，把状态转移、工具调用、轨迹、reward 和评估全部控制住。后面环境接口稳定后，可以再把 PageStore 换成真实 DOM snapshot 或 Playwright observation。

### Q2：这个 expert 是不是作弊？

答：

> expert 是有意设计的 oracle policy，它用于生成 SFT demonstrations，不用于最终评估模型能力。训练数据阶段需要高质量示范，所以 expert 使用任务生成时计算出的 expert_clicks。真正评估时，Base/SFT/GRPO 模型只能根据 instruction 和 observations 自己生成 tool calls。

### Q3：当前任务会不会太简单？

答：

> 当前是 V0/V1 数据闭环，任务集中在单页查找、筛选、排序和简单多步跳转。这样做是为了先验证工具协议和数据管线。后续会扩 difficulty level，比如多页面跳转、多个候选项比较、租房和论文搜索页面，以及 hard task 上的分层评估。

### Q4：SFT 数据为什么保存 messages，而不是只保存最终答案？

答：

> 因为这个项目训练的是 agentic tool use，不是单轮 QA。模型需要学习每一步如何根据 observation 选择下一个 tool call，所以必须保留 user、assistant tool_call、tool observation 的完整多轮 messages。

### Q5：后续 GRPO 的 reward 怎么设计？

答：

> 我会从 rule-based verifier 派生多维 reward。最小版本包括 tool call format reward、final answer reward、step penalty 和 invalid action penalty。进一步可以加入 path reward，比如点击是否沿着合理页面路径前进。这样 reward 既能鼓励最终答案正确，也能减少无效点击和格式错误。

### Q6：怎么评价 SFT 是否有效？

答：

> 我会固定 200 条 eval tasks，对比 Base 和 SFT。核心指标是 tool call format accuracy、task success rate、invalid tool call rate、average step count。预期 SFT 首先会明显提升 tool call 格式稳定性，然后提升一部分任务成功率。

### Q7：为什么要记录 average step count？

答：

> 因为 agent 不只是要答对，还要高效完成任务。如果模型循环点击或走很多多余步骤，即使最后答对，作为 agent 也不是好策略。average step count 后续也可以进 reward，作为 step penalty。

### Q8：invalid action 指什么？

答：

> 当前主要指环境无法执行的工具动作，比如点击当前页面不存在的 element_id、没有打开页面就 click，或者未来 parser 解析不到合法工具名/参数。invalid action rate 是衡量 agent 工具使用稳定性的关键指标。

### Q9：现在 SFT 数据有什么质量保障？

答：

> 第一，expert trajectory 来自可计算 expert path，成功率是 100%。第二，构造 SFT 数据时会校验 assistant 的 tool call 必须是合法 JSON，并且包含 name 和 arguments。第三，所有 episode 都有 summary，可以过滤 failure 或 invalid action 样本。

### Q10：如果模型输出自然语言而不是工具调用怎么办？

答：

> rollout parser 会把这类输出记为 tool call format error 或 invalid tool call。SFT 阶段的一个核心目标就是降低这种错误；GRPO 阶段可以继续通过 format reward 惩罚自然语言输出。

## 9. 当前局限

当前实现有几个明确局限：

- 页面实体数量还小，shopping/course 各 10 个对象。
- 任务模板有限，虽然能生成 1000 条，但本质是模板重采样和轻量 paraphrase。
- expert path 是任务生成时直接给出的，还没有独立搜索策略。
- 环境 observation 目前是纯文本，不包含 DOM tree 或视觉信息。
- verifier 目前只有 exact match，还没有 path reward 和 reward breakdown。
- 还没有接真实模型 rollout、LoRA SFT 和 GRPO。

这些不是问题，而是后续路线：

```text
扩大页面实体和任务模板
-> 接入模型 rollout parser
-> LoRA SFT
-> Base vs SFT eval
-> reward function
-> GRPO
-> Base vs SFT vs GRPO
-> error analysis
```

面试里主动说局限会显得更可信：

> 当前阶段我刻意没有把复杂度铺太开，而是先保证环境和数据管线闭环。下一步我会扩任务难度和网页类型，并开始训练模型，用 eval 指标验证 SFT 和 GRPO 是否真的提升了工具调用稳定性和任务成功率。

## 10. 下一步工作计划

最推荐的下一步：

1. 写 rollout parser，让模型输出的 `<tool_call>` 能被解析和执行。
2. 写 base model eval runner，先不训练，测试 base model 在 200 条 eval 上的工具格式错误率。
3. 写 LoRA SFT 脚本，使用 `training/sft_train.jsonl` 训练 Qwen2.5-0.5B-Instruct 或 Qwen3-0.6B。
4. 跑 SFT eval，对比 Base vs SFT。
5. 实现 reward function，为 GRPO 做准备。

这里的关键里程碑是：

```text
Base model tool call 很不稳定
SFT model tool call format accuracy 明显提升
SFT + GRPO 在 success rate 或 invalid action rate 上进一步提升
```

## 11. 简历描述当前阶段

当前阶段可以弱写成数据和环境建设，但最好等 SFT 或 GRPO 出结果后再正式放进简历。

当前可用描述：

> 构建本地可验证 WebNav 环境，设计 `open_page`、`click`、`get_visible_text`、`submit_answer` 四类工具接口，使 agent 通过多步工具调用完成商品检索和课程查询任务；实现任务生成器、rule-based expert 和 exact-match verifier，自动生成 1000 条 tool-use trajectories，并转换为 800/200 的 SFT train/eval chat 数据。

等完成 SFT 后可以扩展为：

> 基于 expert trajectories 对 Qwen 小模型进行 LoRA SFT，使模型学习结构化 tool-call 协议和多步网页导航流程，并在固定 eval set 上对比 Base/SFT 的 tool call format accuracy、task success rate、invalid action rate 和 average step count。

等完成 GRPO 后可以写成最终版本：

> 设计 rule-based verifier 和多维 reward function，将工具格式、路径正确性、最终答案、无效操作和步数惩罚转化为奖励信号，基于 GRPO 进行 Agentic RL 后训练，并对比 Base/SFT/SFT+GRPO 在网页导航任务上的行为变化。
