# 重装不好用 —— 从根本上解决

> 问题原文：「这会本质上是重装不好用的问题，根本上解决」
>
> 结论先说：**这不是一个 bug，是三个互相叠加的失效，而且是层层掩盖的。**
> 上面那个修好了，下面那个才看得见；下面那个修好了，根因才浮出来。下面按发现顺序写。

---

## 一、三个失效，按「谁掩盖谁」排

| # | 失效 | 症状 | 掩盖了什么 |
|:--|:--|:--|:--|
| 1 | **部署命令不加 `--upgrade`** | 重装跑完，版本还是 `0.4.2` | 让半补丁状态*看起来*像正常的重装 |
| 2 | **半补丁（half-patch）** | 文件都在，功能少一半 | 让 pip 的「删一半」看起来无害 |
| 3 | **pip 在沙箱里被中途杀掉** | `~ore`、`~pu` 这类垃圾留在包里 | —— |

第 1 条是**真正的根因**，前两轮都没发现它，因为手里没有能报告版本的东西。

---

## 二、失效 1：重装是个静默的空操作

`deploy_autoforge.py` 第 77 行原来是：

```python
[VENV_PY, "-m", "pip", "install", "-q", "--no-index",
 "--find-links", WHEELS_DIR, "autoforge-agent"]
```

本机 `wheels/` 里同时有 `autoforge_agent-0.4.2.whl`（365,451 B）和
`autoforge_agent-0.4.2+local.1.whl`（390,542 B）。PEP 440 里本地段 `+local.1`
比没有的版本**高**，所以 pip 本该选它。

但机器上装的是 stock `0.4.2` 时，这条命令的输出是：

```
Requirement already satisfied: autoforge-agent in ... (0.4.2)
```

**什么都没做。**没有 `--upgrade`，pip 不会为了一个「更高的候选」去动一个已满足的需求。
实测（本机、复现过三次）：

```
$ pip install -q --no-index --find-links <wheels> --no-deps autoforge-agent
$ python -c "import importlib.metadata as m; print(m.version('autoforge-agent'))"
0.4.2                      <-- 本地 wheel 就在同一个目录里，没人碰它

$ pip install --no-index --find-links <wheels> --no-deps --upgrade autoforge-agent
0.4.2+local.1              <-- 加了 --upgrade，一次就对
```

这就是「重装不好用」的字面意思：**重装成功了，装的是旧的那个。**
 dist-info 被换掉、版本号看起来动了、pip 返回 0 —— 而 `tools/exposure.py` 没了。

### 修法

`deploy_autoforge.py` 加 `--upgrade`，并在装完之后**硬校验版本号**，不再相信 pip 的返回码：

```python
result = subprocess.run(
    [VENV_PY, "-m", "pip", "install", "-q", "--upgrade", "--no-index",
     "--find-links", WHEELS_DIR, "autoforge-agent"], ...)
...
if "+local." not in installed:
    print("  !! the stock 0.4.2 wheel was installed, not the patched local build.")
    fail("patched build not installed")
```

---

## 三、失效 2：半补丁，最难发现的那一种

`pip uninstall` **只删旧 RECORD 里列出的文件**。补丁 wheel 做了两件事：

| 类别 | 文件 | 在 stock RECORD 里？ | 重装 stock 后 |
|:--|:--|:--|:--|
| 新增 | `media.py`、`tools/exposure.py` | 否 | **幸存** |
| 修改 | `registry.py`、`core/llm.py`、`agent.py` … 11 个 | 是 | **被还原成 stock** |

结果是：`media.py` 在磁盘上、`import autoforge.media` 成功、`autoforge --help`
正常出、`auto chat` 能跑 —— 而 `ToolRegistry.exposure` 不存在，中断响应没了，
每次请求照旧带 26,504 字符的工具 schema。

**半补丁比坏补丁危险，因为它不报错。**

实测复现（本机）：

```
$ pip install --force-reinstall <stock wheel>
$ python -c "from autoforge.tools.registry import ToolRegistry; print(hasattr(ToolRegistry,'exposure'))"
False
$ python -c "import autoforge.tools.exposure"
ModuleNotFoundError: No module named 'autoforge.tools.exposure'
   (有时是这个文件幸存 —— 取决于上一次装的是哪个 wheel，两种情况都见过)
```

### 修法：一个活在包外面的医生

`install_doctor.py`，14 KB，纯 stdlib，放在 `~/.autoforge/`。

**它必须在包外面。**写成 `autoforge/doctor.py` 的话，stock 重装会把
`autoforge/` 整个删掉 —— 唯一能报告这件事的组件，恰好被这件事本身删了。
这次实测里就撞到过一次：

```
$ auto doctor
usage: autoforge ... {chat,forge,...,tick} ...     <-- 没有 doctor
autoforge: error: invalid choice: 'doctor'
```

因为 stock wheel 里没有带 `doctor` 命令的 `cli.py`。

医生做的事：把磁盘上每个文件和 `autoforge-install-manifest.json`
（73 个文件，每个带 sha256）逐个比。分类报告，不合并：

- `version drift` —— 装的是 `0.4.2`，清单说 `0.4.2+local.1`。**一行就抓住半补丁**
- `stale` —— 文件在，但内容不是这个 build（实测 11 个）
- `missing` —— 文件不在（实测 `media.py`、`tools/exposure.py`）
- `unexpected` —— 磁盘上多出来的
- `pip stashes` —— pip 的 `~` 前缀垃圾，单独一类（它该清，但不该触发重装）

---

## 四、写医生时踩到的四个坑（每个都靠测试抓住）

### 1. 遍历把 pip 垃圾也数进去了

第一次跑，一个健康的安装报出 **676 个 unexpected**。因为 `os.walk(site)`
覆盖整个 site-packages：隔离区目录、`certifi`、`requests`、`pip` 自己，
全都不在这份清单里。

**修法**：只走清单里出现过的顶层目录（`autoforge` 和自己的 dist-info）。
别的分发版本不是这次审计的事。

顺带：`dist-info/RECORD` 每次安装都被 pip 重写，所以它**永远**是 stale。
`INSTALLER`、`REQUESTED`、`direct_url.json`、`WHEEL` 同理。
这五个按后缀排除，并且只排除后缀 —— `autoforge/record.py` 不是 `RECORD`。

### 2. 裁剪掉垃圾目录，垃圾就看不见了

第二次跑还是漏：`audit()` 的 walk 里写着
`dirs[:] = [d for d in dirs if not d.startswith("~")]`。

看起来理所当然 —— 它们不是包内容。**后果是 `autoforge/` 里那 68 个文件、
10 个 `~` 目录，对唯一该报告它们的审计完全不可见**，而 `clear_stashes`
照样会去移它们。删掉这个过滤器，立刻报出 **194 个**真实垃圾文件。

### 3. 一个文件报了七遍

修完 2 之后，`pip stashes: 8`，而目录只有 2 个。`expected` 是**按文件**一条记录，
遍历它、每次走一遍 `top`，73 个文件的清单就把包走了 70 多遍。

**修法**：顶层目录去重后再走。

### 4. `shutil.move(src, dst)` 在 dst 已存在时是「移进去」

`--repair` 跑第二次：

```
shutil.Error: Destination path '.../~utoforge_agent-.../~utoforge_agent-...' already exists
```

`quarantine_move()` 改成带计数器的唯一目标名，重复修复幂等，且永不覆盖。

---

## 五、pip 垃圾的形态比想象中乱

同一个 dist-info，本机观察到两种 stash 名：

```
~utoforge_agent-0.4.2+local.1.dist-info     <-- 掩掉 1 个字符
~-toforge_agent-0.4.2.dist-info             <-- 掩掉 2 个字符
```

所以识别标记不能用 `utoforge`，得用 **`toforge`** —— 它在所有变体里都活着
（`~ertifi`、`~-b`、`~ore` 都不含它）。

垃圾还会出现在**任意深度**，不止包顶层：

```
autoforge/~ore                    包内一层
autoforge/tools/~-re              包内两层
autoforge/tools/~%pycache__       __pycache__ 的 stash
```

所以 `clear_stashes` 用 `rglob` + 深的先移，`audit` 的 walk 也按**目录**记一条
（`__pycache__` stash 里只有 `.pyc`，全被跳过，目录本身是唯一能报告它的东西）。

---

## 六、现状

### 本机 install（已全部验证）

```
$ auto doctor
  site-packages : C:\autoforge\venv\Lib\site-packages
  manifest      : 0.4.2+local.1
  installed     : 0.4.2+local.1
  VERDICT       : OK — this install is the build it should be.
```

包树 **0 个 `~` 条目**，67 个 `.py`，与清单逐字节一致。
垃圾统一隔离在 `D:\Users\yangyian.zhang\AppData\Local\autoforge\pip-stash-quarantine`
（**不在 site-packages 里**，一个名叫 `~ore` 的目录本来也不该在 sys.path 上）。

### 端到端（`test_deploy_install_step.py`，非破坏性重放部署第 2 步）

```
1. install stock 0.4.2（模拟坏重装）
     -> version drift 0.4.2 vs 0.4.2+local.1
     -> stale files 12 + missing 3
     -> VERDICT: DRIFTED

2. deploy_autoforge.py 第 2 步（加了 --upgrade）
     -> pip show version: 0.4.2+local.1
     -> VERDICT: OK        <-- 不需要 --repair，一次就对

3. doctor --repair（只清垃圾）
     -> VERDICT: OK, rc=0
```

### 测试

| 套件 | 数量 | 结果 |
|:--|--:|:--|
| `test_install_doctor.py` | 28 | OK |
| `test_exposure_backport.py` | 35 | OK |

`build_wheel.py` 的构建期自检也加了一条：解包 wheel、只从解包目录 import，
断言 `doctor` 在 `build_parser()` 的 choices 里、`command_table()["doctor"]` 可调用。
**少了 CLI 这条钩子的 wheel 会构建失败** —— 否则唯一能报警的通道悄悄没了。

---

## 七、已推到 GitHub（2026-10-16 14:05）

| 东西 | 状态 |
|:--|:--|
| branch `local/reinstall-kit` | `53c4bc59649f`，`local/` 下 **14 个文件** |
| release `local-0.4.2.1`（prerelease） | wheel 资产 **390,542 B**，sha256 `4b3dbd7e20889197` |
| tag `local-0.4.2.1` | 指向 branch 同一个 commit |

新增进 kit 的五个：`install_doctor.py`、`test_install_doctor.py`、
`autoforge-install-manifest.json`、`doctor.cmd`、`test_deploy_install_step.py`。
`install_doctor.py` 和 manifest 都做过字节比对，GitHub 上和本地一致。

### 第五个坑：`github.com` 本体在这台机器上不可达

代理对 `github.com` 的 CONNECT 直接回 **502**。所以「从 GitHub URL 一键重装」
这句话在本机**跑不通**：

```
pip install https://github.com/.../releases/download/local-0.4.2.1/...
→ ProxyError: Tunnel connection failed: 502 Bad Gateway
```

`api.github.com` 通。所以 `download_wheel()` 改成两条路：

1. 先试 manifest 里记录的 `github.com` URL（别的机器上这才是对的）
2. 失败就走 API：先 `GET /releases/tags/{tag}` 拿 asset id（每次重传都会变），
   再 `GET {asset.url}` 且**必须带** `Accept: application/octet-stream`
   —— 少这个头 GitHub 回的是一个 redirect *对象* JSON，不是字节

两条路都要过 sha256 校验：哈希对不上的 wheel 不是这个 build，不管它叫什么。

实测（半补丁状态 + 本地 wheel 全部藏起来，只留 API 这一条路）：

```
verdict: drifted | counts: {'missing': 5, 'stale': 11}
  release url: URLError: Tunnel connection failed: 502 Bad Gateway
  obtained via github api (390,542 B, sha256 ok)
  pip ok
verdict: ok | counts: {'missing': 0, 'stale': 0} | version 0.4.2+local.1
```

顺带：匿名调 API 有 **60 次/小时**限额，超了报 `HTTP 403 rate limit exceeded`。
带 `GITHUB_TOKEN` 是 5000。要可靠就走 token。

### 往返已实测

从 GitHub API 拉下来的 wheel 和本地 `dist/` 里的**逐字节相同**，
直接 `pip install --force-reinstall` 装回去：

```
version     : 0.4.2+local.1
exposure    : True      media : True
abort poll  : 0.25      auto doctor : True
VERDICT     : OK
34 + 35 + 18 测试全绿
```

---

## 八、常用命令

```bash
# 审计（exit 1 = 有漂移）
"C:/autoforge/venv/Scripts/python.exe" -m autoforge.cli doctor

# 审计 + 修（有真漂移才重装，只清垃圾不重装）
"C:/autoforge/venv/Scripts/python.exe" -m autoforge.cli doctor --repair

# 不依赖 autoforge 包本身，任何状态下都能跑
"D:/Users/yangyian.zhang/WorkBuddy AI/2026-10-15-12-51-18/autoforge-deploy/doctor.cmd"

# 让启动时那一行警告闭嘴
AUTOFORGE_NO_DOCTOR=1

# 重装（本机用本地 wheel；github.com 被代理挡，别用那个 URL）
"C:/autoforge/venv/Scripts/python.exe" -m pip install --upgrade --no-index \
  --find-links "D:/Users/yangyian.zhang/WorkBuddy AI/2026-10-15-12-51-18/autoforge-deploy/wheels" \
  autoforge-agent
```

---

## 九、文件清单

| 文件 | 作用 |
|:--|:--|
| `install_doctor.py` | 审计器。纯 stdlib，在包外 |
| `test_install_doctor.py` | 34 个测试，覆盖分类规则、作用域、隔离、下载路由、CLI 钩子 |
| `dist/autoforge-install-manifest.json` | 73 个文件的 sha256 清单 |
| `build_wheel.py` | 从 stock wheel 建补丁 wheel + manifest + 构建期自检 |
| `publish_github.py` | 发 branch + release（需 token） |
| `../2026-10-15-12-51-18/autoforge-deploy/deploy_autoforge.py` | 加 `--upgrade` + 版本硬校验 + 部署时放医生 |
| `../2026-10-15-12-51-18/autoforge-deploy/doctor.cmd` | 一条命令的审计，任何状态可用 |
| `../2026-10-15-12-51-18/autoforge-deploy/test_deploy_install_step.py` | 端到端：坏重装进，OK 出 |

「重装不好用」这句话，最后的答案是三件事，缺一不可：

1. **部署命令得能真的装上新版**（`--upgrade`）—— 否则后面两件都碰不到
2. **得有人能发现装错了**（医生 + manifest）—— 否则半补丁永远藏着
3. **发现的人不能在会被删掉的地方**（包外的独立脚本）—— 否则它自己也活不下来
4. **拉备份的路得能真的拉到**（API 兜底）—— 否则记在册上的 URL 只是好看的
