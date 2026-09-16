# Duplicate needs, from the ledger

Source: `C:\Users\china\AppData\Local\autoforge\autoforge.db` -- 653 recorded needs, 210 distinct texts, 181 clusters.

- **20 clusters are the same job under different tool names** (23 names). These are real duplicates: the merge is keep the best, retire the rest.
- **145 clusters are the same need retried under one name.** The tool exists and the need came back, so the defect is retrieval, not the shelf.

## Merge these: same job, different names

keep is the one with the most events, or the clearest name when tied

### 3 names, 10 forge events, 3 phrasings

  names: append_gitignore_rules, restore_gitignore_files, restore_gitignore_original

  - 把 autoforge 和 tool-market 两个仓的 .gitignore 恢复成我修改前的原始内容（去掉我追加的 *.bak、prom*.txt、%SystemDrive%/、toolmarket.db 等规则）
  - 往 autoforge 和 tool-market 两个仓的 .gitignore 追加规则：*.bak、prom*.txt、%SystemDrive%/、toolmarket.db，保留原有内容
  - 把 autoforge 仓的 .gitignore 写回原始内容（不含 *.bak、*.bak.*、prom*.txt 三行），把 tool-market 仓的 .gitignore 也写回原始内容（不含 *.bak、*.bak.*、%SystemDrive%/、toolmarket.db 等行），然后打印两个文件确认

### 2 names, 15 forge events, 6 phrasings

  names: autoforge_bus_cli, autoforge_bus_send_read

  - Run the autoforge agent-bus CLI: register as autoforge, list boards, list who, and read the autoforge board with --peek --everything so I can see all 12 message
  - Run the autoforge bus subcommand correctly: `autoforge bus register --as autoforge`, `autoforge bus boards`, `autoforge bus who`, `autoforge bus read --as autof
  - Run the autoforge bus CLI correctly: `autoforge bus send --as autoforge --board autoforge --to hermes-ef3ad8 "text"` then `autoforge bus read --as autoforge --b

### 2 names, 11 forge events, 3 phrasings

  names: grep_editor_defs, grep_py_defs

  - Find the terminal line editor class in the autoforge package (the object passed as `editor` to _LiveRun, with .available, .write, .tick methods) — grep all .py 
  - Grep every .py file under D:\Users\china\Desktop\项目_开发\autoforge\autoforge for the string 'def tick' and 'def write' and print file, line number and the line, t
  - Search all .py files under D:\Users\china\Desktop\项目_开发\autoforge\autoforge for the definition of the terminal line editor class: find lines containing 'def tic

### 2 names, 7 forge events, 2 phrasings

  names: inspect_config_json, read_config_keys_masked

  - 打印 C:\Users\china\.autoforge\config.json 的完整结构（api_key 打码），特别是 notify 段的嵌套层级
  - 读 C:\Users\china\.autoforge\config.json 的内容，打印所有键（api_key 打码）

### 2 names, 6 forge events, 2 phrasings

  names: autoforge_ledger_sessions, autoforge_sessions

  - Read the autoforge sqlite ledger at C:\Users\china\AppData\Local\autoforge\autoforge.db and print distinct session ids with event counts and time ranges, to tel
  - Read the autoforge sqlite ledger at C:\Users\china\AppData\Local\autoforge\autoforge.db: print the schema of forge_events, then group events by whatever session

### 2 names, 6 forge events, 2 phrasings

  names: git_show_file_context, git_show_file_snippet

  - 在 D:\Users\china\Desktop\项目_开发\autoforge 目录下运行 git show HEAD:autoforge/core/lineedit.py，打印其中 _visible_len 函数附近的 20 行，看 HEAD 版本里这个函数长什么样
  - 在 D:\Users\china\Desktop\项目_开发\autoforge 目录下运行 git show HEAD:autoforge/core/lineedit.py，把输出保存到临时文件，然后打印包含 _visible_len 的那一段（前后各15行）

### 1 names, 15 forge events, 4 phrasings

  names: srcpatch

  - crlf_patch(path, old, new): replace exactly one occurrence of old with new in a text file on a Windows host, adapting to the file's own line endings. Steps: rea
  - crlf_srcpatch(path, old, new): same as srcpatch but line-ending agnostic. Read the file bytes and decode utf-8. Normalize the file text by replacing '\r\n' with
  - crlfpatch(path, old, new): patch a utf-8 text file regardless of its line endings. Read bytes, decode utf-8, normalize '\r\n' to '\n', count occurrences of old 

### 1 names, 11 forge events, 3 phrasings

  names: git_repo_status

  - 检查 autoforge 源码目录 D:\Users\china\Desktop\项目_开发\autoforge 是不是 git 仓库，打印 git status 和 remote
  - 在 D:\Users\china\Desktop\项目_开发\autoforge 目录下运行 git status、git branch、git remote -v、git diff --stat，返回仓库状态
  - 检查 autoforge 和 toolmarket 两个仓的 git 状态：有没有未提交改动、有没有未推送的 commit、remote 是什么，用于判断是否需要同步到 github

### 1 names, 7 forge events, 2 phrasings

  names: find_import_source

  - 读 autoforge/agent.py 前 80 行的 import 段，找 load 是从哪个模块导入的
  - 在 autoforge/agent.py 里找 load 函数的 import 来源，确定它读哪个配置文件路径

### 1 names, 7 forge events, 2 phrasings

  names: restore_gitignore

  - 把 autoforge 仓的 .gitignore 写回原始内容（不含我追加的 *.bak、prom*.txt 规则），并确认 tool-market 的 .gitignore 也恢复原样
  - 用 python 把 autoforge 仓的 .gitignore 恢复成原始内容（去掉我追加的 *.bak、*.bak.*、prom*.txt 三行），并打印恢复后的内容确认

### 1 names, 7 forge events, 2 phrasings

  names: toolstore_audit

  - toolstore_audit(action)：盘点 toolmarket 货架与本地工具库，全部靠真实 IO，读不到就 ok=false 并报错误，绝不编造。
签名 toolstore_audit(action, limit=20)：
action='market' → 用 urllib GET 环境变量 TOOLM
  - toolstore_audit(action)：盘点 toolmarket 货架与本地工具库，全部靠真实 IO，读不到就 ok=false 并报错误，绝不编造。
签名 toolstore_audit(action, limit=20)：
action='market' → 用 urllib GET 环境变量 TOOLM

### 1 names, 6 forge events, 2 phrasings

  names: toolmarket_transition_probe

  - 探测 toolmarket transition 端点的真实请求体字段：发空 JSON 与错误 action，读 422 校验错误详情
  - 对 toolmarket 的 transition 端点做真实探测：用真实资源 id 试 POST，读 404/405/422 的响应体，判断端点是否真的存在

### 1 names, 6 forge events, 2 phrasings

  names: toolmarket_restore_tool

  - 用 Python 直接 POST toolmarket transition 端点，把 tool:toolmarket_list 从 retired 恢复为 active，并报告服务端是否允许
  - 用 Python 直接 POST toolmarket transition 端点，把资源从 retired 恢复为 active（或报告服务端拒绝的原因）

### 1 names, 6 forge events, 2 phrasings

  names: patch_lineedit_wcwidth

  - 备份 D:\Users\china\Desktop\项目_开发\autoforge\autoforge\core\lineedit.py 到同目录 lineedit.py.bak，然后把 _visible_len 函数改成用 wcwidth 计算显示宽度（中文占2列），保留 ANSI 码和控制字符不计列的原有行为，并在
  - 用 Python 精确替换 lineedit.py 里的 _visible_len 函数：把 return sum(1 for ch in _ANSI.sub("", text) if ch >= " ") 换成用 wcwidth 计算宽度的实现，并在文件顶部 import 区加入容错 wcwidth 导入。改完打印替

### 1 names, 5 forge events, 2 phrasings

  names: read_agent_bus

  - Read the hermes agent-bus file mailbox at %LOCALAPPDATA%\hermes\agent-bus (autoforge.ndjson) and print its raw contents plus the directory listing, so I can see
  - Read the hermes agent-bus mailbox at C:\Users\china\AppData\Local\hermes\agent-bus (autoforge.ndjson): print directory listing and raw file contents

### 1 names, 5 forge events, 2 phrasings

  names: run_lineedit_pytest

  - 在 D:\Users\china\Desktop\项目_开发\autoforge 目录下运行 pytest 跑 lineedit 相关测试，返回通过/失败结果
  - 在 D:\Users\china\Desktop\项目_开发\autoforge 目录下运行 pytest，只跑跟 lineedit 相关的测试，返回结果

### 0 names, 8 forge events, 2 phrasings

  names: 

  - 本机多 agent 协调总线：基于文件锁的互斥（claim/release 资源，防止两个 agent 同时改同一 store 或推同名工具）+ 基于追加日志的消息广播（send/inbox，让本机所有 agent 实例互相通信）。纯标准库，跨进程安全，Windows 友好
  - 本机多 agent 协调总线：文件锁互斥（claim/release 资源名，防止两个 agent 同时改同一 store 或推同名工具）+ 追加日志消息广播（send/inbox，让本机所有 agent 实例互相通信）。纯标准库，跨进程安全

### 0 names, 8 forge events, 2 phrasings

  names: 

  - source_read(path, start=None, end=None, contains=None)：如实读取磁盘上一个文本文件并把读到的字节原样报告出来，不做任何解释、总结或推测。必须返回这些字段：ok(布尔)、path(绝对路径)、size(字节数)、sha256(原始字节的哈希)、line_count、t
  - source_read(path, start=None, end=None, contains=None)：如实读取磁盘上一个文本文件并把读到的内容原样报告，不解释、不总结、不推测。返回 JSON 字段：ok(布尔)、path(绝对路径)、size(字节数)、sha256(原始字节哈希)、line_count、tot

### 0 names, 6 forge events, 2 phrasings

  names: 

  - 重启 toolmarket 服务：杀掉监听 8000 的 python 进程，用 C:\Users\china\toolmarket_server.py 重新拉起，等待端口就绪并验证
  - 重启 toolmarket 服务：杀掉监听 8000 的 python 进程，用 C:\Users\china\toolmarket_server.py 重新拉起，并等待端口就绪

### 0 names, 4 forge events, 2 phrasings

  names: 

  - patch_lines(path, start, end, lines): replace an inclusive 1-based line range in a utf-8 text file, joining the replacement lines with the file's own NEWLINE st
  - apply_lines(path, start, end, newlines): rewrite an inclusive 1-based line range of a utf-8 text file. `newlines` is a list of strings (one per replacement line

## Look at retrieval, not at the shelf

- 4x  Read the hermes artifacts on this host: C:\Users\china\chrome-hermes, hermes-optimized-prompts.md, hermes-skills-backup.tar.gz, install_hermes.sh — an
- 4x  列出本机正在运行的 python/autoforge 进程，并检查是否存在共享的 store 文件或监听端口，以判断是否有另一个 agent 实例在线
- 4x  列出本机所有 python 进程的完整命令行（用 wmic 或 powershell），并检查 8000 端口上监听的是什么进程，以确认是否有另一个 autoforge agent 实例在运行
- 4x  用 powershell 列出本机 python 进程的 PID 和命令行，并查出 8000 端口由哪个 PID 监听
- 4x  读取本机 agent-bus 文件板：返回 agents.json 内容、boards 目录下各板的行数、以及指定板最后 N 条消息的解析结果（作为函数返回值，不依赖 stdout）
- 4x  读取 agent-bus 文件板并返回内容：agents.json 全文、各板行数、指定板最后 N 条消息
- 4x  向 agent-bus 文件板注册自己并发送消息：以 autoforge 会话身份 register（写 agents.json）和 send（追加一行 ndjson 到 boards/autoforge.ndjson），纯标准库，原子追加
- 4x  判断本机 agent 会话是否真的在线：读取 agent-bus 的 agents.json 拿到各会话的 session id 和启动时间，再用 tasklist/wmic 列出活着的 hermes.exe 和 auto.exe 进程及其启动时刻，比对得出每个会话 online/offline，不
- 4x  Find every autoforge agent instance on this host: list all python processes with full command lines via wmic/tasklist, and read the autoforge sqlite l
- 4x  List all running python processes with their full command lines on this Windows host using wmic process where "name like '%python%'" get ProcessId,Com
- 4x  Run tasklist and return its raw output, to see which python processes are alive on this host
- 4x  Liveness check for peer agents: given the agent-bus agents.json registry and the live process table, report for each registered agent whether it is AL
- 4x  列出 toolmarket 上的资源清单并读取若干条目，看有哪些工具、状态字段（是否已晋升/审核）
- 4x  直接调用 toolmarket 的 transition 端点做一次真实的状态迁移（如 promote），并报告返回码与响应体
- 4x  用 Python 直接请求 toolmarket 的 openapi.json 与 /resources/{id}/transition 的 405/422 响应，拿到 transition 端点的真实字段要求
- 4x  统计 toolmarket 上 83 个资源的状态分布（draft/promoted 等），并列出未晋升资源的 id
- 4x  统计 toolmarket 资源状态分布并列出未晋升条目（只取 id/name/state 字段，避免大响应）
- 4x  在本机找到 toolmarket 服务的源码目录（含 transition 状态机、retired 终态定义的文件）
- 4x  读取 toolmarket POST /resources 的 409 冲突响应体，看服务端拒绝重名注册的具体原因
- 4x  列出本机正在运行的 python 进程及其命令行，找到 toolmarket 服务进程的 pid

## Method, and what it cannot see

Cluster when two needs share a tool name, or their token sets overlap by Jaccard >= 0.45, or a >25-character need is a substring of a longer one: 210 needs, 106 distinct tool names, 165 clusters flagged. Name-sharing is the strongest signal and the substring rule is the weakest; a chain of the three can put two needs in one cluster that no single rule would join.

Not seen: a tool that was renamed. Two names for one job with no shared vocabulary and no substring relation land in different clusters, and the fix for that is not a better threshold -- it is a rename recorded as an event, which the ledger does not yet do.