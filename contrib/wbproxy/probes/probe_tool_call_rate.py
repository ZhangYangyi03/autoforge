"""How stable is the trigger probe across models?

A forge round failed with `trigger: fired 0/1` and the identical need then passed
on retry. That is not a connection problem -- the proxy answered 200 throughout --
it is the model declining to call a tool it was just given. Since autoforge's
trigger check is `hits > 0` over `trigger_trials=1` by default, a single "no" is
a whole failed round.

This measures the rate directly: one tiny tool, one clear request, N trials per
model. The point is to choose a default for `--model` that does not cost the user
a re-forge every third attempt.
"""
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8777/v1/chat/completions"

TOOL = [{"type": "function", "function": {
    "name": "count_words",
    "description": "Count the number of whitespace-separated words in a text string.",
    "parameters": {"type": "object",
                   "properties": {"text": {"type": "string"}},
                   "required": ["text"]}}}]

PROMPT = "count the words in the sentence 'hello world foo bar'"
TRIALS = 5


def one(model: str) -> tuple[str, str]:
    body = {"model": model, "max_tokens": 512, "temperature": 0.0,
            "tools": TOOL, "messages": [{"role": "user", "content": PROMPT}]}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            doc = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return "error", f"http {e.code}"
    except Exception as e:                              # noqa: BLE001
        return "error", f"{type(e).__name__}"
    calls = doc["choices"][0]["message"].get("tool_calls") or []
    if calls:
        return "called", calls[0]["function"].get("name") or "?"
    return "silent", (doc["choices"][0]["message"].get("content") or "")[:50]


def main() -> int:
    models = sys.argv[1:] or ["hy3", "step-5-preview", "deepseek-v4.1-flash",
                              "glm-5.3-flash"]
    print(f"tool-call rate over {TRIALS} trials each, temperature 0\n")
    print(f"{'model':26s} {'called':>7s}  outcomes")
    print("-" * 74)
    for model in models:
        outcomes = []
        for _ in range(TRIALS):
            kind, detail = one(model)
            outcomes.append(kind)
        called = outcomes.count("called")
        print(f"{model:26s} {called}/{TRIALS:<5d}  {','.join(outcomes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
