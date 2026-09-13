# autoforge — 设计立场

> 一个从零造起的 agent 框架。核心主张：**工具即数据，自造的工具必须挣得被调用的权利。**
> 自由度是默认，验证是可挂载的能力，不是笼子。

本文记录框架从主流 agent 系统里**取了什么精华、扔了什么糟粕**，每一条设计决策都可回溯到出处。

---

## 一、精华 / 糟粕 对照总表

| 来源系统 | 取的精华 | 扔的糟粕 |
|:---|:---|:---|
| **Hermes Agent** | skills 作为程序性记忆、跨会话持久记忆、curator 生命周期（stale→archive→backup）、provider 无关、profiles 隔离、cron/kanban 耐用调度 | 工具变更需 `/reset` 才生效（为 prompt caching 牺牲热插拔）；skill 是文档不是可执行代码；无触发验证 |
| **Claude Code** | worktree 隔离、subagent 委派、hooks、checkpoint/rollback、项目上下文文件、todo 追踪、权限模式（plan/accept/yolo） | 闭源、单 provider 绑定；权限模式对工具**创建**本身无治理 |
| **Codex CLI** | 沙箱化执行（seatbelt/landlock）、审批模式（suggest/auto-edit/full-auto） | 沙箱靠 OS 特性，跨平台不一致；无工具生命周期 |
| **OpenHands (CodeAct)** | **动作即代码**而非 JSON tool call — 表达力强、可组合、可复用为函数 | 全量代码进上下文 → 上下文膨胀；无工具去重 |
| **Voyager** | **可执行 skill 库** + 自动课程 + 执行反馈迭代 + **入库前自我验证**（关键！） | skill 检索仅靠 embedding 文本相似度，不关心执行效果 |
| **Reflexion** | 失败后自我反思 → 存为教训（lesson），下次注入 | 纯文本反思，无结构化、无验证闭环 |
| **MCP** | 标准化工具接口；tools / resources / prompts 分离；动态发现 | stdio/HTTP 传输层复杂度；无质量语义 |
| **LangGraph** | 显式状态图、checkpoint、durable execution、human-in-the-loop interrupt | 对简单任务过度工程；图定义冗长 |
| **AutoGPT / BabyAGI** | 任务分解 + 自主循环 | 无限循环无终止条件；无验证；无记忆卫生 |
| **Tea Agent / ATLASS** | 运行时工具生成 + 闭环工具学习；执行前安全检查 | 难生成需外部依赖的复杂工具；触发问题未解 |
| **Memento-Skills** | **行为对齐路由**（按执行成功率检索，而非文本相似度） | 依赖离线 RL 训练路由器，重 |
| **Ouroboros** | 修改自身实现代码、审查过的提交（reviewed commit） | 过于激进，改动核心实现的爆炸半径大 |
| **ICLR'26 Misevolution** | 明确「误进化」失败模式：工具创建/复用中引入漏洞或退化 | （这是被引以为戒的问题，不是要学的东西） |

---

## 二、由此得出的七条架构决策

### 1. 工具即数据，热插拔（反 Hermes 的 `/reset` 妥协）
registry 持有 schema 列表，新工具下一轮即可用。prompt caching 的代价我们选择用**分级暴露**来付：只有 ACTIVE 工具进上下文，DRAFT/QUARANTINED 默认不进。→ 取自 Hermes 的分层思想，但拒绝它的重启要求。

### 2. 工具带合同出生：schema + 触发探针 + 效果签名（反 Hermes/Voyager 的「只有文档」）
`ToolSpec` 不只是函数，是「一个带契约的声明」。合同让工具可被自动验证。→ 概念取自 MCP 的 interface 标准化，但把「质量语义」补上。

### 3. 生命周期是状态机，不是布尔值（反 Claude Code 的权限二值）
`DRAFT → PROBATION → ACTIVE → QUARANTINED → RETIRED`。免费创建（PROBATION 立即可调用），但 ACTIVE —— 那个挣到上下文预算和信任的状态 —— 必须靠通过探针、然后靠战绩**保持**。→ 取自 Hermes curator 的生命周期思想，但作用对象从「技能文档」升级为「可执行工具」。

### 4. 触发验证是一等公民（直击 Constraint Tax）
验证分两个正交问题：
- **触发问题**：该调用它时，Agent 会不会调用？（Constraint Tax 的病灶）
- **执行问题**：调用了，结果对不对？

主流框架几乎只测后者。我们两个都测，且触发测试用**正例 + 负例**（会误触发的工具比从不触发的更糟）。→ 直接来自本次对话定义的瓶颈一。

### 5. 沙箱限爆炸半径，不封能力上限（反 Codex/OpenHands 的白名单式限制）
进程隔离 + 超时 + 环境清洗，但**不阉割 builtins**。自由度是用户要的。

**要紧的一条，写清楚免得误读**：这个沙箱**不把 agent 与宿主机隔开**。锻造出的代码是本进程的子进程，跑在**同一台机器**上，有**完整的宿主机文件系统**和**出站网络** —— 「没有共享文件系统 / 没有共享进程空间 / 没有网络通道」是**错的**。它护住的是 agent 的循环（`os._exit`／段错误／死循环打不垮主进程，刷屏不污染协议），不是这台机器，也不是对外的安全边界。

正因为射程如此，agent 对自身能力的自述必须是**测量**而不是散文：`Sandbox.reach()` 返回结构化事实，`reach(probe=True)` 跑一次真实往返（在 cwd 之外写读一个文件 + 解析域名）作为证据；`my_capabilities` 直接渲染它，`forge_tool` 与 `my_capabilities` 的工具描述里也写明 —— 描述每回合都随请求下发，比提示词里的一段话更难被忽略。「我没有文件工具」是错的：锻造一个就有。→ 取 Codex 的进程隔离精华，扔它跨平台不一致的糟粕（Windows 用 timeout 兜底）。

### 6. 战绩账本 + 自动隔离（反 Misevolution）
每个工具调用都记账。成功率跌破阈值或连续失败达限 → 自动 QUARANTINED（从上下文隐藏，但 `force=True` 仍可跑）。可 rehab 回 PROBATION。→ 来自 ICLR'26 误进化论文定义的问题，Memento-Skills 的行为对齐路由是其检索侧的兄弟方案。

### 7. 动作即代码，但代码是资产（取 CodeAct，扔其上下文膨胀）
工具是 Python 函数，可被组合、可被复用、可被序列化进库。但**不是所有代码都进上下文** —— 只有 ACTIVE 的 schema 进。→ CodeAct 的表达力 + 分级暴露的克制。

---

## 三、与现有框架的核心差异（一句话）

| 框架 | 它的核心问题 |
|:---|:---|
| Hermes | 技能是文档，不可执行、无触发验证 |
| Claude Code | 工具集固定，权限管的是「调用」不是「创建」 |
| CodeAct/OpenHands | 动作即代码，但代码全量进上下文、无质量治理 |
| Voyager | 有 skill 库和入库验证，但检索不看执行效果 |
| Tea/ATLASS | 会造工具，但触发问题未解、退化无检测 |
| **autoforge** | **造得出 + 触得发 + 用不坏 —— 三段闭环** |
