# 分布式节点：让另一台机器真的能连上这台

为什么有这些文件：autoforge 之前只能被"它自己所在的那台机器"驱动。所谓分布式，
第一步就是这台机器上有一个**别的机器能到达**的执行端点——不是 127.0.0.1，是真的 TCP。
2026-09-20 在 Windows 机上跑通并实测。

## 组成

    node_server.py   端点服务。纯标准库 http.server，不依赖 venv/uvicorn/fastapi。
                     理由：它必须能在什么都没装的机器上以裸进程启动；一个带依赖清单的
                     中转服务，恰好在最需要它的时候起不来。
    node_client.py   另一端用的客户端。在 KOS Linux 机上直接 `python3 node_client.py ...`。
    node.token       共享密钥（0600）。所有会改状态的接口都要 Bearer token。
    DISTRIBUTED.md   本文件。

## 为什么每个写接口都要 token

把 /exec 无鉴权地绑在 0.0.0.0 上，就是把一个远程代码执行端点挂到局域网上，还写着我的名字。
/health 开放，是因为需要鉴权的健康检查没法给监控用；其余全部要 token。
坏 token 实测返回 401。

## 接口

    GET  /node/health              开放。存活 + 这台机器是什么
    GET  /node/info                token。路径、工具、本机 toolmarket 可达性
    POST /node/exec                token。{"cmd": "shell 行 或 [argv]", "timeout": 120, "cwd": ...}
    POST /node/submit              token。{"code": "...", "task": "名字"} -> 作业 id（异步）
    GET  /node/job/{id}            token。状态 + 结果
    GET  /node/jobs                token。列表
    GET|POST /market/<path>        toolmarket 反向代理（见下）

作业状态落在 jobs/*.json，不在内存里：和 toolmarket 用 sqlite 而不是 :memory: 同一个理由——
第一次重启吃掉货架，就是这条教训的代价。

## 两个容易踩的坑（都实测过）

1. **字符串命令必须走 shell，列表必须不走。** 把 `"echo ok && python x.py"` 这种字符串
   用 shell=False 交给 subprocess，Windows 会拿整行当文件名去找，报 WinError 2。
   现在：str = shell 行（shell=True），list = argv（shell=False）。

2. **HTTP/1.1 + 每次 `Connection: close` 会让前置代理 502。** 协议版本写 1.1 就是在承诺
   keep-alive，而这个服务发完就关；代理下次把第二个请求放到已经关掉的 socket 上，
   于是得到一个"没有回应的回应"。症状是间歇性的，所以难查。改成 HTTP/1.0 后消失。

## ngrok 只做一件事：绕过 127.0.0.1

toolmarket 的启动器**故意**只绑回环（它的注释写着：公开面交给 ngrok，绑 0.0.0.0 就有两条
入口，而只有一条受隧道的限流保护）。所以"连上 toolmarket"在别的机器上不是改一行 host，
是把这个回环面搬到公网。免费套餐只给一条 HTTP 隧道，所以：

> **这条"只绑回环"是本仓库那个启动器的行为，不是 toolmarket 的性质。** 2026-09-21 由 KOS 那台
> 机器实测纠正：他们那边移植后的启动器绑的是 `0.0.0.0:8000`，所以从这台机 `curl
> http://192.168.1.108:8000/health` **直接通，不需要 token、不需要隧道**。我先前把本机的规则
> 当成通用规则递了过去，对方按"必须走隧道"去排查，白找一圈。
>
> 判据不要靠文档，靠一条命令：在**对端**跑 `netstat -ltn | grep 8000`（Windows：
> `netstat -an | findstr :8000`）。看到 `0.0.0.0:8000` 就是局域网直达，看到 `127.0.0.1:8000`
> 才是必须借道本节点或隧道。同一份代码，两台机器可以是两种绑定，因为启动器是移植时各自改的。

- ngrok 的一条隧道指向**节点**（localhost:8077）
- 节点把 toolmarket 反向代理在 `/market/*` 下

一条 URL 带两半：`/node/*` 是执行端点，`/market/*` 是货架。

实测：12/12 成功率。**注意**：曾经有一个旧隧道还指向早已死掉的 8023 端口，
ngrok 把请求轮询到那个死后端，于是 502 是间歇性的（实测 5/12）。
删掉旧隧道后变成 12/12。排查任何"有时通有时不通"的时候，先看 4040 上还有几条隧道。

## 从 KOS Linux 机连过来

端口 8077 在 LAN 上本来就通（Windows 防火墙三个配置文件都是关的，实测），
所以同网段时不需要隧道：

    python3 node_client.py health  --url http://<这台机的局域网 IP>:8077
    python3 node_client.py exec "nproc" --url http://<这台机的局域网 IP>:8077
    python3 node_client.py submit-file job.py --name nightly

token 放 $AUTOFORGE_NODE_TOKEN，或放在 node_client.py 旁边。
连不通时它**非零退出**——静默不等于成功。

不同网段（KOS 那个 10.173.47.130 从这里 ping 不通）走公网 URL：

    curl https://<你的 ngrok 域名>/node/health
    curl -H "Authorization: Bearer $TOKEN" https://<域名>/node/exec -d '{"cmd":"nproc"}'

## 没有做成的事，以及为什么

- **ssh 走 ngrok**：ngrok 免费套餐的 TCP 端点要绑卡（实测 502 + 明确提示），
  而 HTTP 隧道拒绝 CONNECT（实测 421 Misdirected Request）。所以 ssh-over-tunnel 这条路
  在这个账号上是关的。要走 ssh 就得同网段，或者换 tailscale/zerotier 这类打洞方案。
- **到 KOS 机的隧道**：KOS 自己没有可达的公网面，从这里 ping 10.173.47.130 也 100% 丢包，
  两边不在同一个二层网。所以现在是"KOS 主动连过来"的星型结构，不是对等网状。
- **WSL 只验证到进程级**：WSL2 现在是 mirrored 网络模式（和 Windows 共用 192.168.1.107），
  它对 8077 的连接被同一个 TCP 栈自连回去，不算跨栈验证。真正的跨栈验证是上面那条
  ngrok 公网路径——请求确实换了一个 IP、一个 TLS 终止点才回到这台机。

## Federating two shelves (both machines' tools in one lookup)

Two machines each with a shelf is not one market until each one's forge
*before* it builds asks the other. That is `AUTOFORGE_PEER_MARKETS`:

    setx AUTOFORGE_PEER_MARKETS "kos=http://192.168.1.108:8000"

Comma-separated `label=url`. A peer reached through the node's authenticated
proxy takes its token from a file rather than the environment:

    setx AUTOFORGE_PEER_MARKETS "win=http://192.168.1.107:8077/market|FILE:C:\Users\china\autoforge_node\node.token"

Set on THIS machine already: `kos=http://192.168.1.108:8000`.
For the KOS machine to use this one's 6953 tools it sets the `win=...` line
above, with the token file copied over.

What the lookup does with a peer, and why each rule is the way it is:

  peer answered, carries a hit     reported first, labelled with the peer name --
                                   a peer hit is not a local hit, the tool has to
                                   be called *there*
  peer answered, carries nothing   an ANSWER: licenses the forge. The verdict says
                                   the local half is missing when it was, so one
                                   machine's answer never reads as the whole market's
  peer switched off                NOT an answer. A machine being down must not read
                                   as an empty world -- and a peer that just failed is
                                   skipped for two minutes, so a dead peer costs one
                                   timeout, not one per forge (measured 4.10s then 0.00s)
  no peers configured              byte-for-byte the same requests as before the
                                   feature existed. A LAN sweep on every forge would
                                   not be a lookup, it would be a scan
