"""Task-to-tool retrieval: colloquial operator task -> reusable tool names.

Ported/derived from three open-source projects (all stdlib-only here):

  * rank_bm25 (dorianbrown/rank_bm25, Apache-2.0): BM25Okapi idf with its
    epsilon floor and the k1/b length normalisation. Numpy and multiprocessing
    dropped; the corpus here is a few hundred short descriptions.
  * RAKE-tutorial (zelandiya/RAKE-tutorial, MIT): word score = degree/frequency,
    used to pick the concrete words out of a colloquial task. Candidate phrases
    are approximated with adjacent-token windows because a task is one sentence.
  * LangChain MultiQueryRetriever (langchain-ai/langchain, MIT): several views
    of one query, unique union, best score wins. The LLM rewriter is replaced
    by deterministic rewrites so no upstream model call is needed.

The Chinese verb/action table and stop-word list are local additions measured
on this host's operator phrasing 2026-11-13. Chinese stop words partly derived
from goto456/stopwords (MIT). External HTTP retrieval with its abstain gate is
left to toolmarket's own /search endpoint; this module ranks locally.

Why this exists: on 2026-11-13 the operator's real task "传一下刚才做好的最新版
autoforge去github..." hit 0 tools through the old term-overlap candidate
injector and abstained on the shelf, while its canonical paraphrase hit@1.
Colloquial verbs (传/推/搞/看一下), filler (刚才/最新版/是不是好了) and
secondary intent (拿这个实践一下) diluted the query. This module expands the
verbs into the vocabulary tool descriptions actually use and drops the filler.
"""

import re, math
from collections import Counter
from functools import lru_cache

# ---- domain vocabulary (local additions) ----
CN2EN = {
 "读取":"read", "查看":"read", "查询":"query", "获取":"get", "额度":"quota",
 "配额":"quota", "用量":"usage", "余额":"balance", "页面":"page", "网页":"page",
 "登录":"login", "账号":"account", "调用":"invoke", "工具":"tool", "货架":"market",
 "探测":"probe", "端口":"port", "连通":"reachability", "网络":"network",
 "检测":"check", "测试":"test", "状态":"status", "合并":"merge", "通知":"notify",
 "配置":"config", "邮件":"email", "发送":"send", "文件":"file", "缩放":"resize",
 "图片":"image", "水印":"watermark", "压缩":"compress", "裁剪":"crop",
 "转换":"convert", "格式":"format", "秒数":"seconds", "时间":"time",
 "转成":"format", "生成":"generate", "分析":"analyze", "统计":"stats",
 "列表":"list", "删除":"delete", "创建":"create", "修改":"patch", "修复":"fix",
 "重启":"restart", "数据库":"database", "索引":"index", "资源":"resource",
 "检索":"search", "搜索":"search", "命中":"hit", "结果":"results",
 "日志":"log", "进程":"process", "提交":"commit", "推送":"push", "远程":"remote",
 "仓库":"repo", "内容":"content", "打印":"print", "本地":"local",
}
# colloquial verb -> tool-description vocabulary (the gap measured 2026-11-13)
ACTION = {
 "传":"commit push upload transfer", "上传":"commit push upload", "推":"commit push",
 "推送":"commit push", "提交":"commit push", "发":"send notify", "交":"commit push",
 "搞":"fix patch build", "弄":"fix patch", "修":"fix patch repair", "改":"patch modify",
 "看":"read check status inspect", "查":"read check query inspect", "瞧":"read check",
 "瞅":"read check", "念":"read", "列":"list", "跑":"run test execute",
 "试":"test verify measure", "实践":"test verify measure", "测":"test measure",
 "拿":"use test", "用":"use", "装":"install", "删":"delete remove", "清":"delete remove",
 "干掉":"delete remove", "整理":"fix clean", "查证":"verify", "核实":"verify",
 "看一下":"read check status", "查一下":"read check query", "瞅一眼":"read check",
}
STOP_ZH = set("的 了 着 过 吗 呢 啊 吧 把 按 用 上 中 为 与 和 或 及 页 一个 一下 可以 如何 怎么 什么 是否 需要 进行 以及 相关 对应 这个 那个 我们 你们 他们 它 这 那 有 在 刚才 做好的 最新版 是不是 好了 为什么 咋样 帮我 请 拿 这个 一下 现在 还 就 也 都 又 再 更 最 很 太 挺 一下 把 让 给 从 到 去 拿 一下 这个 那个 里 上 下 前 后 时候 之后 之前".split())
STOP_EN = set("""a an and any are as at be been but by can could did do does for from
get had has have how i if in into is it its of on or our so that the their
them then there these this to use used using was we were what when where which
who why will with would you your please help me my now just also about after
before new latest current today good make made done""".split())
WORD = re.compile("[a-z0-9_]+")
CJK = re.compile("[\u4e00-\u9fff]+")

def tokenize(text, action_expand=False):
    text = str(text).lower()
    out = []
    for w in WORD.findall(text):
        if w in STOP_EN or len(w) < 2: continue
        out.append(w)
        for part in w.split("_"):
            if len(part) >= 2 and part not in STOP_EN: out.append(part)
    for run in CJK.findall(text):
        # Chinese bigrams, minus function words
        for i in range(len(run)-1):
            bg = run[i:i+2]
            if bg not in STOP_ZH: out.append(bg)
        for zh, en in CN2EN.items():
            if zh in run: out += [x for x in WORD.findall(en) if x not in STOP_EN]
        if action_expand:
            for zh, en in ACTION.items():
                if zh in run: out += [x for x in WORD.findall(en) if x not in STOP_EN]
    return list(dict.fromkeys(out))

# ---- rank_bm25 BM25Okapi, numpy-free port (Apache-2.0) ----
class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.doc_freqs = [Counter(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / float(self.corpus_size or 1)
        nd = Counter()
        for d in corpus:
            nd.update(set(d))
        idf_sum, negative = 0.0, []
        self.idf = {}
        for w, f in nd.items():
            v = math.log(self.corpus_size - f + 0.5) - math.log(f + 0.5)
            self.idf[w] = v; idf_sum += v
            if v < 0: negative.append(w)
        eps = self.epsilon * (idf_sum / len(self.idf) if self.idf else 0.0)
        for w in negative: self.idf[w] = eps
    def get_scores(self, query):
        s = [0.0] * self.corpus_size
        # rank_bm25 applies an epsilon floor only to *negative* idf values.
        # Measured 2026-11-13 on a 2-document regression docset, applying it
        # to every term instead makes common words score 0 and yields no hits.
        avg_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0
        floor = self.epsilon * avg_idf
        for q in query:
            if q not in self.idf:
                continue
            idf = self.idf.get(q)
            if idf <= 0:
                idf = max(floor, 1e-6)
            for i, freq in enumerate(self.doc_freqs):
                f = freq.get(q, 0)
                if not f: continue
                s[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
        return s

# ---- RAKE deg/freq word scores (MIT), used to pick concrete task words ----
def rake_scores(tokens):
    freq, degree = Counter(tokens), Counter()
    # 2-word windows stand in for RAKE candidate phrases
    for a, b in zip(tokens, tokens[1:]):
        degree[a] += 1; degree[b] += 1
    return {w: (degree[w] + freq[w]) / float(freq[w]) for w in freq}

# ---- MultiQueryRetriever union (MIT): several views, unique union, best score ----
def query_views(task):
    t = str(task).strip()
    views = []
    def add(v):
        v = v.strip()
        if v and v not in views: views.append(v)
    add(t)
    ids = [w for w in WORD.findall(t.lower()) if len(w) >= 2 and w not in STOP_EN]
    if ids: add(" ".join(ids))
    # action-expanded canonical view: what the tool description would say
    canon = tokenize(t, action_expand=True)
    canon_words = [w for w in canon if re.fullmatch(r"[a-z0-9_]+", w)]
    if canon_words: add(" ".join(canon_words))
    # A short, all-English, high-precision view for the shelf's coverage gate.
    # Chinese bigrams and filler terms lower coverage and make /search abstain.
    acts = [w for w in canon_words
            if w in ("commit", "push", "upload", "transfer", "read", "check",
                     "status", "query", "inspect", "list", "run", "test",
                     "fix", "patch", "delete", "send", "notify", "parse",
                     "measure", "verify", "generate", "analyze", "search")]
    short = " ".join(list(dict.fromkeys(ids + acts))[:8])
    if short: add(short)
    # For the shelf's coverage gate, a very short query of just the action
    # words plus the top identifier is the highest-precision view. Measured
    # 2026-11-13: "github commit push" hits git_add_commit_push at rank 1,
    # while views carrying extra task terms drop below the 0.60 coverage cut.
    if acts and ids:
        for ident in ids[:2]:
            add((" ".join([ident] + acts[:3])).strip())
    # concrete keyword view (RAKE top words)
    toks = tokenize(t, action_expand=False)
    rs = rake_scores(toks)
    key = [w for w, _ in sorted(rs.items(), key=lambda x: -x[1])[:12] if w not in STOP_ZH]
    if key: add(" ".join(key))
    return views

def rank(task, docs, limit=5):
    corpus = [tokenize(text) for _, text in docs]
    bm = BM25Okapi(corpus)
    best = [0.0] * len(docs)
    for view in query_views(task):
        q = tokenize(view, action_expand=True)
        for i, s in enumerate(bm.get_scores(q)):
            if s > best[i]: best[i] = s
    # A short, concrete task can share only one discriminating term with its
    # tool (e.g. "git repository status branch" -> "git repo status ...").
    # rank_bm25's epsilon floor keeps such terms from being discarded, so do
    # not reintroduce an arbitrary overlap-count cut here.
    order = sorted(range(len(docs)), key=lambda i: (-best[i], docs[i][0]))
    return [(docs[i][0], round(best[i], 6)) for i in order[:limit] if best[i] > 0]
