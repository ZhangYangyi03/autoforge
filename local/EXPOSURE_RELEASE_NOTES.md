# exposure 回移交付说明

**日期**：2026-10-16
**目标**：修好「安装版 0.4.2 缺 schema exposure 机制」——即每次请求都把全部工具 schema 塞进 prompt 的问题。

---

## 1. 问题

部署的 `autoforge_agent-0.4.2` wheel 早于上游的 exposure 分层机制。没有它，agent 循环会把**每一个可见工具**的完整 schema 序列化进**每一次**模型调用，永久如此。

在这台机器上实测：

| | 数量 | 字符 |
|:--|--:|--:|
| 可见 schema | 59 | 26,504 |
| 其中 7 天内调用过的 | 14 | — |

也就是说，45 个**一周内一次都没被调用**的工具，它们的完整参数表在每一次请求里都要重新付一遍 token。

---

## 2. 移植了什么

上游 `main` 的 exposure 机制，逐块搬进 `C:\autoforge\venv\Lib\site-packages\autoforge\`。共 5 个文件：

| 文件 | 改动 | 说明 |
|:--|:--|:--|
| `tools/exposure.py` | **新增** 6,502 B | 冷热分层的全部逻辑。与上游逐字节一致（md5 `2e9c9ccaa323`） |
| `tools/registry.py` | +5,637 B | `last_called_source` / `window_days` / `protect` / `_calls_seen` / 缓存；`days_since_called()`、`exposure()`；`call()` 里加调用计数 |
| `core/agent.py` | +3,284 B | `Agent` 接 `exposure=` 参数；新增 `_request()`——只把**热** schema 当 tools 发，冷的作为一条消息注入在 system prompt 之后 |
| `store.py` | +1,339 B | `last_called_by_tool()`——从 `forge_events` 账本读「每个工具最后一次调用时间」 |
| `agent.py` | +8,179 B | `TOOL_SCHEMA_BUDGET_CHARS`、`_tool_schema_chars()`、`_tool_budget_report()`、`_hot_protect()`、`_last_called_at()`；以及三处接线 |

**上游设计的三个要点**（照搬，未改）：

1. **按行为分，不按数量分。** 判据是「7 天内有没有被调用过」，不是「取前 N 个」。硬性数量上限需要一个 tie-break，而任何 tie-break（名字序、成本、年龄）都会因为与「是否被需要」无关的理由降级某个工具。
2. **冷工具没有被隐藏。** 它以「名字 + 参数名 + 一行描述」的形式留在 prompt 里，随时可调用，`describe_tool(name)` 拿回完整 schema。降级一行 20 字符，却消掉了「工具存在但模型不知道」这一整类失败。
3. **HOT_FLOOR 是地板不是偏好。** `run_python` / `forge_tool` / `describe_tool` / `list_tools` / `remember` / `recall` / `terminate` 永不降级——它们是模型在还不知道自己要什么时会伸手去拿的那几个。

另外两个上游刻意的细节，也一并搬了：

- **`protect`**：本次任务文本里**按词边界**出现的工具名保持热。账本回答的是「最近用了什么」，而任务可能涉及账本从没见过的东西。
- **缓存键含小时**：`exposure()` 的结果缓存到「发生一次调用」或「整点」为止。缓存让两次**没有调用**的连续轮次构建出**逐字节相同**的 prompt 前缀——这正是 provider 前缀缓存所需要的。小时在键里是因为时间本身也会改变答案（7 天前调用过的工具，第 8 天必须自己变冷）。

---

## 3. 实测结果

### 静态测量（真实账本，150 个工具被调用过，其中 19 个在 7 天内）

```
可见 schema   : 59
完整 payload  : 26,504 字符  (~6,626 tokens @4字符/token)
热（完整下发）: 14 个 schema, 6,771 字符
冷（一行清单）: 45 行,       5,691 字符
分流后合计    : 12,462 字符  (~3,115 tokens)
```

### 红检（把唯一那行接线拆掉再测）

红检脚本 `redcheck_exposure.py` 的关键是**每次测量跑在独立进程里**——`autoforge.agent` 每个解释器只导入一次，在同一个进程里改磁盘文件什么都不会变。第一版脚本正是栽在这里，两次测的是同一个缓存模块，于是「证明」了分流是摆设。

| 状态 | 完整 schema | 冷清单 | 合计 |
|:--|--:|--:|--:|
| 接线在 | 15 个 / 7,139 | 5,986 | **13,125** |
| 接线拆掉 | 60 个 / 26,872 | 0 | **26,872** |
| 恢复后 | 15 个 / 7,139 | 5,986 | **13,125** |

```
PASS: 26,872 -> 13,125 chars per request; 13,747 saved (51.2% cut).
      60 schemas in full -> 15 in full + 45 as one line each.
```

**每次请求省 51.2%**。剩下的 5,986 字符是 45 条冷清单本身的开销——这是上游「一行一个工具」的设计代价，没有动。

> 注意：这里 `wired` 是 15 个热、`static` 测量里是 14 个。差的那一个是任务文本命中了 `protect`（红检用 "hello there"，静态测量用默认构造）——两个数都是对的，取的是不同任务。

---

## 4. 测试

新增 `test_exposure_backport.py`，**35 个测试全绿**。覆盖：

- **分层本身**（6）：地板永不降级、没记录就是冷、7.0 天算热 / 7.01 天算冷、`protect` 与 `floor` 不混、冷行确实只有一行且截断可见、manifest 文案里说了「可调用」
- **registry**（7）：没账本且没记录返回 `None` 而非 0、调用计数递增、**被拦下的调用不算使用**（否则工具会靠它没做过的调用保持热）、缓存同一对象、调用后缓存失效、**账本优先于 spec.stat**、账本抛异常时回退不炸
- **账本查询**（4）：每个工具取最新、只有 `call` 事件算数、坏 payload 不炸查询、空账本返回空
- **请求构造**（7）：无 exposure 时全发且不注入、有 exposure 时只发热、冷工具注入在 system prompt 之后且任务是最后一条、**注入块绝不进入 transcript**（否则 resume 会把它当成用户说的话重放）、回调炸了回退到全发、没有冷工具就不注入、**两次无调用轮次前缀逐字节相同**
- **ForgeAgent 接线**（7）：有 store 时绑定账本、裸 registry 保持 None、`_hot_protect` 只按整词匹配（`seen` 不保护 `see`）、多词名要连续 token 段（`browser goto` 命中、`goto the browser` 不命中）、空任务不保护、`protect` 用完即清、预算报告只算 agent 自己的工具、**agent 确实把 registry 交给了循环**
- **真实数据**（2）：payload 确实变小、floor 工具在真实数据上确实是热的

原有 `test_autoforge_interrupt.py` **18 个测试仍然全绿**。

---

## 5. 安装 / 回滚

```bash
cd "D:/Users/yangyian.zhang/WorkBuddy AI/2026-10-16-10-08-39/autoforge-interrupt-fix"

# 看会做什么，不动文件
C:/autoforge/venv/Scripts/python.exe install_exposure.py --dry-run

# 安装（重装 wheel 之后跑这个把回移放回去）
C:/autoforge/venv/Scripts/python.exe install_exposure.py

# 回滚到原状
bash rollback-exposure.sh
```

`install_exposure.py` 的防呆：目标文件必须是**原始 0.4.2** 或**已打补丁**（按 md5 比对 `rollback-exposure/` 和 `patched/`），两者都不是就拒绝覆盖（除非 `--force`）。重装 wheel → `install_exposure.py` → 回滚 → 再装，这个循环已实测走通且幂等。

---

## 6. 文件清单

| 文件 | 用途 |
|:--|:--|
| `patched/` | 5 个修补后文件的完整快照（安装脚本的载荷） |
| `rollback-exposure/` | 4 个原始文件的备份（回滚载荷） |
| `exposure.diff` | 统一 diff，611 行，新增 488 行——人读用 |
| `install_exposure.py` | 安装 / 校验 / 幂等 |
| `rollback-exposure.sh` | 回滚 |
| `test_exposure_backport.py` | 35 个测试 |
| `redcheck_exposure.py` | 红检（独立进程测量 + 拆线对比 + 恢复确认） |

---

## 7. 还没做的

- **PR #1（多模态）仍未合并**：https://github.com/ZhangYangyi03/autoforge/pull/1。这次的回移**没有**进 PR——exposure 是上游 `main` **已有**的东西，`main` 上早就有了。这次只是把安装版补齐到上游水平，**不是新贡献**，所以没有提 PR 的理由。
- **工具调用仍不可中断**：`registry.call` 是同步的，这是 store 里记录的候选修复「C」。
- **`forge_events` 没有 pid / instance 列**：同一 db 上跑两个 `auto` 进程无法区分。
- **exposure 只治了 schema 那一半**。token 账单的大头在**前缀缓存**：网关实测缓存命中 ¥0.02/M vs 未命中 ¥1/M（**50 倍**）。exposure 的缓存设计正是为了让无调用轮次的前缀逐字节稳定——但真正的省钱要靠长会话里前缀不变，这需要实际跑起来才看得到。
