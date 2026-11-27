import pytest
from autoforge.browser_login import login_step, GMAIL

class Fake:
    def __init__(self): self.urls=[]
    def goto(self, url, **kw): self.urls.append(url)
    def evaluate(self, expression): return {"status": "unknown"}

def test_gmail_navigation():
    b=Fake(); assert login_step(b,"gmail")["status"] == "unknown"; assert b.urls == [GMAIL]

def test_wechat_requires_explicit_https():
    b=Fake()
    for url in ("", "http://example.com", "https://user:pw@example.com"):
        with pytest.raises(ValueError): login_step(b,"wechat",url)
    assert not b.urls

def test_wechat_navigation():
    b=Fake(); login_step(b,"wechat","https://example.com/login"); assert b.urls == ["https://example.com/login"]

def test_reject_unknown_action():
    with pytest.raises(ValueError): login_step(Fake(),"other")
