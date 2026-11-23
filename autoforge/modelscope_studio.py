"""modelscope_studio: run a ModelScope xGPU studio's Gradio API with a studio_token.

Measured 2026-11-23 on this host:
- each studio is served at https://<owner>-<name>.ms.show (owner/name lowercased, /  -> -)
- /gradio_api/info and /config are readable without login; /gradio_api/call/* and
  /gradio_api/queue/join are 403 without a studio_token query param
- the token is issued when a logged-in browser opens the studio page; it appears in the
  iframe URL as ?studio_token=<uuid>
Usage:
    from modelscope_studio import studio_token_from_browser, queue_call
    tok = studio_token_from_browser("major-tom-earthembeddingexplorer")
    print(queue_call("Major-TOM", "EarthEmbeddingExplorer", tok, 23, ["glacier",7,"FarSLIP",False,False,"2016-01-01","2024-12-31",False,-90,90,-180,180]))
"""
import json, re, urllib.request, uuid

CDP = "http://127.0.0.1:9222/json/list"

def host_of(owner, name):
    return "https://%s-%s.ms.show" % (owner.lower(), name.lower().replace(".", "-"))

def get(url, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout).read()

def studio_token_from_browser(substr):
    """Read the studio_token out of the iframe URL of an already-open, logged-in studio page."""
    for t in json.load(urllib.request.urlopen(CDP, timeout=10)):
        if substr in t.get("url", ""):
            m = re.findall(r"studio_token=([0-9a-f\-]{36})", t["url"])
            if m:
                return m[-1]
    raise RuntimeError("no studio_token for %r - open the studio page in the logged-in browser first" % substr)

def queue_call(owner, name, token, fn_index, args, timeout=300):
    """Call one Gradio fn over the queue API and return the completed output dict."""
    host = host_of(owner, name)
    sh = uuid.uuid4().hex[:11]
    payload = {"data": args, "event_data": None, "fn_index": fn_index,
               "trigger_id": 18, "session_hash": sh}
    req = urllib.request.Request(host + "/gradio_api/queue/join?studio_token=" + token,
                                 data=json.dumps(payload).encode(),
                                 headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
                                 method="POST")
    urllib.request.urlopen(req, timeout=60)
    raw = get(host + "/gradio_api/queue/data?session_hash=%s&studio_token=%s" % (sh, token), timeout).decode("utf-8", "ignore")
    done = None
    for chunk in raw.split("\n\n"):
        if chunk.startswith("data: "):
            try:
                d = json.loads(chunk[6:])
            except Exception:
                continue
            if d.get("msg") == "process_completed":
                done = d
    if done is None:
        raise RuntimeError("no completion: " + raw[-300:])
    return done
