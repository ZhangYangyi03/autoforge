"""Reaching a person, and being honest about whether it worked.

The modules under test are the agent's only way to say anything to anybody who
is not sitting in front of its stdin. Two properties matter more than the rest
and get most of the coverage here:

  * a failure is *reported*, never swallowed. A send that quietly does nothing
    is worse than a send that was never attempted, because the caller believes
    somebody was told.
  * a credential is never echoed. A delivery report ends up in a transcript, a
    log, a screenshot, or a bug report.

The webhook tests run against a real socket on localhost rather than a mocked
urlopen, so what is exercised is the actual urllib path -- status handling,
error classes, and the difference between "answered 500" and "could not
connect" all come from the library, not from a stand-in that agrees with us.
"""

from __future__ import annotations

import http.server
import smtplib
import threading
from email.message import EmailMessage

import pytest

from autoforge.notify import (
    EMAIL,
    WEBHOOK,
    Channel,
    Notifier,
    NotifyError,
    channels_from_config,
)

# ----------------------------------------------------------------------
# a real HTTP endpoint on localhost
# ----------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    """Records what arrived, and answers with whatever the test asked for."""

    received: list[dict] = []
    status = 200

    def do_POST(self):                                   # noqa: N802 - stdlib name
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", "replace")
        type(self).received.append({"path": self.path, "body": body,
                                    "headers": dict(self.headers)})
        self.send_response(type(self).status)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_GET(self):                                    # noqa: N802 - stdlib name
        self.do_POST()

    def log_message(self, *a):                           # silence the test output
        pass


@pytest.fixture
def endpoint():
    """A live localhost HTTP server; yields (url, handler class)."""
    _Handler.received = []
    _Handler.status = 200
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/hook", _Handler
    finally:
        server.shutdown()
        server.server_close()


def webhook_channel(url: str, **kw) -> Channel:
    return Channel.from_dict({"kind": WEBHOOK, "name": "ops", "url": url, **kw})


# ----------------------------------------------------------------------
# reading the config
# ----------------------------------------------------------------------


class TestChannelFromDict:

    def test_webhook_needs_a_url(self):
        with pytest.raises(NotifyError, match="has no url"):
            Channel.from_dict({"kind": WEBHOOK})

    def test_webhook_url_must_be_a_url(self):
        with pytest.raises(NotifyError, match="must start with"):
            Channel.from_dict({"kind": WEBHOOK, "url": "ftp://example.com/x"})

    def test_unknown_kind_is_named(self):
        with pytest.raises(NotifyError, match="kind must be one of"):
            Channel.from_dict({"kind": "carrier-pigeon"})

    def test_missing_kind_is_named_as_missing(self):
        with pytest.raises(NotifyError, match="missing"):
            Channel.from_dict({"url": "https://example.com"})

    def test_email_needs_host_and_recipient(self):
        with pytest.raises(NotifyError, match="no host"):
            Channel.from_dict({"kind": EMAIL, "to": "a@b.c"})
        with pytest.raises(NotifyError, match="no 'to' address"):
            Channel.from_dict({"kind": EMAIL, "host": "smtp.example.com"})

    def test_a_single_recipient_string_becomes_a_list(self):
        ch = Channel.from_dict({"kind": EMAIL, "host": "h", "to": "a@b.c"})
        assert ch.to == ["a@b.c"]

    def test_port_defaults_follow_tls(self):
        assert Channel.from_dict(
            {"kind": EMAIL, "host": "h", "to": "a@b.c", "tls": "ssl"}).port == 465
        assert Channel.from_dict(
            {"kind": EMAIL, "host": "h", "to": "a@b.c"}).port == 587

    def test_sender_falls_back_to_user(self):
        ch = Channel.from_dict({"kind": EMAIL, "host": "h", "to": "a@b.c",
                                "user": "me@b.c"})
        assert ch.sender == "me@b.c"

    def test_headers_are_stringified(self):
        ch = webhook_channel("https://example.com/h", headers={"X-N": 3})
        assert ch.headers == {"X-N": "3"}

    def test_non_dict_headers_rejected(self):
        with pytest.raises(NotifyError, match="headers must be an object"):
            Channel.from_dict({"kind": WEBHOOK, "url": "https://e.com", "headers": "x"})


class TestRedaction:
    """A password must not survive a config dump."""

    def test_to_dict_never_carries_the_password(self):
        ch = Channel.from_dict({"kind": EMAIL, "host": "h", "to": "a@b.c",
                                "user": "u", "password": "hunter2"})
        dumped = str(ch.to_dict())
        assert "hunter2" not in dumped
        assert ch.to_dict()["password"] == "(set)"

    def test_absent_password_is_distinguishable_from_set(self):
        ch = Channel.from_dict({"kind": EMAIL, "host": "h", "to": "a@b.c"})
        assert ch.to_dict()["password"] == "(unset)"

    def test_describe_does_not_carry_it_either(self):
        ch = Channel.from_dict({"kind": EMAIL, "host": "h", "to": "a@b.c",
                                "user": "u", "password": "hunter2"})
        assert "hunter2" not in ch.describe()


class TestChannelsFromConfig:

    def test_no_notify_block_is_not_an_error(self):
        assert channels_from_config({}) == ([], [])
        assert channels_from_config(None) == ([], [])
        assert channels_from_config({"notify": {}}) == ([], [])

    def test_a_bad_entry_is_reported_and_the_rest_survive(self):
        """One unreadable channel must not cost the others."""
        cfg = {"notify": {"channels": [
            {"kind": WEBHOOK, "name": "good", "url": "https://example.com/h"},
            {"kind": WEBHOOK, "name": "broken"},
            {"kind": EMAIL, "name": "mail", "host": "h", "to": "a@b.c"},
        ]}}
        channels, problems = channels_from_config(cfg)
        assert [c.name for c in channels] == ["good", "mail"]
        assert len(problems) == 1
        assert "broken" in problems[0]

    def test_problem_names_its_position(self):
        cfg = {"notify": {"channels": [
            {"kind": WEBHOOK, "name": "good", "url": "https://e.com/h"},
            {"kind": "nope"},
        ]}}
        _, problems = channels_from_config(cfg)
        assert "#2" in problems[0]

    def test_disabled_channel_is_skipped_silently(self):
        cfg = {"notify": {"channels": [
            {"kind": WEBHOOK, "name": "off", "url": "https://e.com/h",
             "enabled": False},
            {"kind": WEBHOOK, "name": "on", "url": "https://e.com/h"},
        ]}}
        channels, problems = channels_from_config(cfg)
        assert [c.name for c in channels] == ["on"]
        assert problems == []

    def test_channels_not_a_list_is_reported(self):
        channels, problems = channels_from_config({"notify": {"channels": "x"}})
        assert channels == []
        assert problems and "not a list" in problems[0]


# ----------------------------------------------------------------------
# deciding to send
# ----------------------------------------------------------------------


class TestSendPreconditions:

    def test_no_channels_refuses_and_says_how_to_fix_it(self):
        with pytest.raises(NotifyError, match="notify.channels"):
            Notifier([]).send("hello")

    def test_empty_text_is_refused(self):
        ch = webhook_channel("https://example.com/h")
        with pytest.raises(NotifyError, match="empty"):
            Notifier([ch]).send("   \n  ")

    def test_an_unmatched_only_list_refuses_and_lists_what_exists(self):
        ch = webhook_channel("https://example.com/h")
        with pytest.raises(NotifyError, match="no channel matches"):
            Notifier([ch]).send("hi", only=["nosuch"])

    def test_bool_and_describe_report_configuration(self):
        assert not Notifier([])
        assert Notifier([webhook_channel("https://e.com/h")])
        assert "No notification channels" in Notifier([]).describe()
        assert "1 channel(s)" in Notifier(
            [webhook_channel("https://e.com/h")]).describe()


# ----------------------------------------------------------------------
# webhook delivery, over a real socket
# ----------------------------------------------------------------------


class TestWebhookDelivery:

    def test_a_200_is_reported_as_delivered(self, endpoint):
        url, handler = endpoint
        out = Notifier([webhook_channel(url)]).send("build finished")
        assert len(out) == 1 and out[0].ok
        assert out[0].status == 200
        assert len(handler.received) == 1

    def test_the_body_carries_the_text_and_a_source(self, endpoint):
        import json
        url, handler = endpoint
        Notifier([webhook_channel(url)]).send("hello there", subject="Subject")
        payload = json.loads(handler.received[0]["body"])
        assert payload["text"] == "hello there"
        assert payload["subject"] == "Subject"
        assert payload["source"] == "autoforge"

    def test_channel_headers_are_sent(self, endpoint):
        url, handler = endpoint
        Notifier([webhook_channel(url, headers={"X-Token": "abc"})]).send("hi")
        assert handler.received[0]["headers"].get("X-Token") == "abc"

    def test_a_500_is_a_failure_that_says_not_delivered(self, endpoint):
        url, handler = endpoint
        handler.status = 500
        out = Notifier([webhook_channel(url)]).send("hi")
        assert not out[0].ok
        assert out[0].status == 500
        assert "NOT delivered" in out[0].detail

    def test_a_dead_port_is_a_failure_not_an_exception(self):
        """The caller gets a verdict; nothing propagates out of send()."""
        ch = webhook_channel("http://127.0.0.1:9/nothing-listens-here",
                             timeout=2.0)
        out = Notifier([ch]).send("hi")
        assert not out[0].ok
        assert "could not reach" in out[0].detail

    def test_one_dead_channel_does_not_stop_the_live_one(self, endpoint):
        """The reason to have two channels is that one may be down."""
        url, handler = endpoint
        notifier = Notifier([
            webhook_channel("http://127.0.0.1:9/dead", timeout=2.0),
            webhook_channel(url),
        ])
        out = notifier.send("hi")
        assert [d.ok for d in out] == [False, True]
        assert len(handler.received) == 1

    def test_duration_is_recorded(self, endpoint):
        url, _ = endpoint
        out = Notifier([webhook_channel(url)]).send("hi")
        assert out[0].duration_ms >= 0.0


# ----------------------------------------------------------------------
# email delivery
# ----------------------------------------------------------------------


class _FakeSMTP:
    """Stands in for the transport so the *logic* is what gets tested."""

    mode = "ok"
    sent: list[EmailMessage] = []

    def __init__(self, host=None, port=None, timeout=None, context=None):
        self.host, self.port = host, port
        self.logged_in: tuple | None = None

    def ehlo(self):
        return (250, b"ok")

    def starttls(self, context=None):
        return (220, b"go")

    def login(self, user, password):
        if type(self).mode == "auth":
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")
        self.logged_in = (user, password)

    def send_message(self, message):
        if type(self).mode == "refuse":
            raise smtplib.SMTPRecipientsRefused({"a@b.c": (550, b"no")})
        type(self).sent.append(message)

    def quit(self):
        return (221, b"bye")

    def close(self):
        pass


@pytest.fixture
def smtp(monkeypatch):
    _FakeSMTP.mode = "ok"
    _FakeSMTP.sent = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)
    return _FakeSMTP


def mail_channel(**kw) -> Channel:
    base = {"kind": EMAIL, "name": "mail", "host": "smtp.example.com",
            "to": ["a@b.c"], "user": "me@b.c", "password": "hunter2",
            "sender": "me@b.c"}
    return Channel.from_dict({**base, **kw})


class TestEmailDelivery:

    def test_a_clean_send_is_delivered(self, smtp):
        out = Notifier([mail_channel()]).send("body text", subject="Hi")
        assert out[0].ok
        assert smtp.sent[0]["Subject"] == "Hi"
        assert smtp.sent[0].get_content().strip() == "body text"

    def test_credentials_are_actually_passed_through(self, smtp):
        Notifier([mail_channel()]).send("x")
        assert smtp.sent, "nothing reached send_message"

    def test_auth_failure_names_the_problem_and_not_the_password(self, smtp):
        smtp.mode = "auth"
        out = Notifier([mail_channel()]).send("x")
        assert not out[0].ok
        assert "rejected the login" in out[0].detail
        assert "hunter2" not in out[0].detail

    def test_refused_recipients_are_named(self, smtp):
        smtp.mode = "refuse"
        out = Notifier([mail_channel()]).send("x")
        assert not out[0].ok
        assert "refused" in out[0].detail

    def test_empty_subject_falls_back_to_the_channel_default(self, smtp):
        Notifier([mail_channel(subject="autoforge")]).send("x")
        assert smtp.sent[0]["Subject"] == "autoforge"


# ----------------------------------------------------------------------
# the account given to the caller
# ----------------------------------------------------------------------


class TestSummary:

    def test_all_failed_says_nobody_was_told(self, endpoint):
        url, handler = endpoint
        handler.status = 503
        notifier = Notifier([webhook_channel(url)])
        summary = notifier.summary(notifier.send("x"))
        assert "NOBODY WAS NOTIFIED" in summary

    def test_partial_delivery_is_counted(self, endpoint):
        url, _ = endpoint
        notifier = Notifier([
            webhook_channel("http://127.0.0.1:9/dead", timeout=2.0),
            webhook_channel(url),
        ])
        summary = notifier.summary(notifier.send("x"))
        assert "1 of 2 channels delivered" in summary

    def test_full_delivery_is_just_the_lines(self, endpoint):
        url, _ = endpoint
        notifier = Notifier([webhook_channel(url)])
        summary = notifier.summary(notifier.send("x"))
        assert "http 200" in summary
        assert "NOT" not in summary

    def test_nothing_sent_is_stated(self):
        assert Notifier([]).summary([]) == "Nothing was sent."
