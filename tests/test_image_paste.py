"""Pasting a picture, not a link to one.

The gesture this covers is a screenshot on the clipboard and Ctrl+V at the
prompt -- no file on disk, no path typed. What makes it work is that the
line editor reads the *picture* formats rather than the text one, writes the
bytes to a file, and leaves a placeholder on the line that the message layer
puts back together with the image when the request is built.

Every test here injects the clipboard reader, because the alternative is a test
that fights the operator for the real clipboard.
"""
from __future__ import annotations

import base64
import io
import re

import pytest
from PIL import Image, ImageDraw

from autoforge.core import lineedit
from autoforge.core.agent import image_parts
from autoforge.core.message import Message
from autoforge.core.steering import Steering


def _png(text: str = "hello", size=(120, 40)) -> bytes:
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).text((6, 12), text, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _nothing_on_the_real_clipboard(monkeypatch):
    """Every test here answers its own clipboard question.

    What is actually on the operator's clipboard is not a fixture. This ran as
    a real failure once: a test patched the picture reader only, so `_paste`
    fell through to the *files* reader, found the folder copied five minutes
    earlier, and attached it.
    """
    monkeypatch.setattr(lineedit, "read_clipboard_files", lambda: [])
    monkeypatch.setattr(lineedit, "read_clipboard", lambda: "")
    monkeypatch.setattr(lineedit, "read_clipboard_png", lambda: b"")


@pytest.fixture
def clip(monkeypatch):
    """A clipboard that holds a picture, and nothing else."""
    def _set(data: bytes):
        monkeypatch.setattr(lineedit, "read_clipboard_png", lambda: data)
        monkeypatch.setattr(lineedit, "read_clipboard", lambda: "")
    _set(_png())
    return _set


def test_a_picture_on_the_clipboard_becomes_a_placeholder(clip, tmp_path):
    editor = lineedit.LineEditor(stream=io.StringIO(), paste_to=tmp_path)
    assert editor._paste() is True
    line = "".join(editor._buf)
    assert line.startswith(lineedit.IMAGE_MARK)
    assert str(tmp_path) in line
    assert editor.take_images() == [line]


def test_the_picture_is_taken_once(clip, tmp_path):
    """A second line must not carry the picture the first one already sent."""
    editor = lineedit.LineEditor(stream=io.StringIO(), paste_to=tmp_path)
    editor._paste()
    assert len(editor.take_images()) == 1
    assert editor.take_images() == []


def test_an_empty_clipboard_pastes_nothing(clip, tmp_path):
    clip(b"")
    editor = lineedit.LineEditor(stream=io.StringIO(), paste_to=tmp_path)
    assert editor._paste() is False
    assert editor._buf == []


def test_text_still_wins_when_there_is_text(clip, tmp_path):
    """A copied *file* carries a path in the text format; that is what a plain
    paste inserts, and turning it into an image attach would be wrong."""
    editor = lineedit.LineEditor(stream=io.StringIO(), paste_to=tmp_path,
                                 clipboard_read=lambda: "C:\\shots\\a.png")
    assert editor._paste() is True
    assert "".join(editor._buf) == "C:\\shots\\a.png"


def test_text_paste_alone_still_works(tmp_path):
    editor = lineedit.LineEditor(stream=io.StringIO(), paste_to=tmp_path,
                                 clipboard_read=lambda: "just words")
    assert editor._paste() is True
    assert editor.take_images() == []


def test_a_picture_rides_with_the_line_it_was_pasted_into(clip, tmp_path):
    line = lineedit.collapse_image(_png(), directory=tmp_path) + "\nwhat is wrong here?"
    parts = image_parts(line)
    assert parts is not None
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert parts[0]["text"] == line
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # ...and base64 of the actual bytes, not of a name.
    decoded = base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1])
    assert decoded == _png()


def test_a_line_with_no_picture_is_still_a_plain_string():
    """The complaint this guards against is a text model being handed a list of
    parts it cannot read: no picture, no parts, the message it always was."""
    assert image_parts("just a question") is None


def test_a_placeholder_whose_file_is_gone_sends_the_text_only(tmp_path):
    line = "[Image #9: 12 bytes \u2192 %s]" % (tmp_path / "vanished.png")
    assert lineedit.image_paths(line) == [str(tmp_path / "vanished.png")]
    assert image_parts(line) is None


def test_the_wire_content_is_parts_and_the_stored_text_is_text(clip, tmp_path):
    line = lineedit.collapse_image(_png(), directory=tmp_path) + "\nlook"
    message = Message.user(line, images=image_parts(line))
    wire = message.to_api()
    assert isinstance(wire["content"], list)
    assert [p["type"] for p in wire["content"]] == ["text", "image_url"]
    # Everything else in the framework reads `.content`: compaction, the ledger,
    # the transcript. It has to stay the text it was.
    assert message.content == line


def test_steering_hands_the_pictures_on(monkeypatch, tmp_path):
    steering = Steering(stream=io.StringIO())
    monkeypatch.setattr(steering.editor, "take_images", lambda: ["[Image #1: 5 bytes \u2192 x.png]"])
    assert steering.take_images() == ["[Image #1: 5 bytes \u2192 x.png]"]


def test_an_editor_without_take_images_is_not_a_crash(monkeypatch):
    """A pipe has no editor. Asking it for pictures has to answer "" rather than
    raise, or a non-interactive run dies at the first prompt."""
    steering = Steering(stream=io.StringIO())
    monkeypatch.setattr(steering, "editor", object(), raising=False)
    assert steering.take_images() == []


def test_collapse_image_names_a_png_that_is_a_png(tmp_path):
    ref = lineedit.collapse_image(_png(), directory=tmp_path)
    path = re.search(r"\u2192 (.+?)\]$", ref).group(1)
    with open(path, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(__import__("os").name != "nt", reason="Win32 clipboard formats")
def test_the_real_clipboard_gives_back_png_bytes(monkeypatch):
    """The one test that touches the operator's clipboard, and the only place
    the DIB header is read for real. It does not read the screen: it puts a
    bitmap on the clipboard first and takes it back."""
    import ctypes
    import ctypes.wintypes as wt

    # The module-level fixture replaced the readers with stubs so that no other
    # test depends on the operator's clipboard. This is the one test whose whole
    # subject *is* the real clipboard, so it puts the real readers back -- and
    # then sets the clipboard itself rather than reading whatever was there.
    monkeypatch.undo()

    image = Image.new("RGB", (90, 30), "white")
    ImageDraw.Draw(image).text((4, 8), "clip", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="BMP")
    data = bytearray(buffer.getvalue()[14:])
    data[24:28] = (3780).to_bytes(4, "little")   # biXPelsPerMeter: 96 dpi

    user32, kernel32 = lineedit._clipboard_api()
    user32.SetClipboardData.argtypes = [wt.UINT, wt.HGLOBAL]
    user32.SetClipboardData.restype = wt.HANDLE
    kernel32.GlobalAlloc.argtypes = [wt.UINT, ctypes.c_size_t]
    assert lineedit._open_clipboard(user32)
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(0x0002, len(data))
        ptr = kernel32.GlobalLock(handle)
        ctypes.memmove(ptr, bytes(data), len(data))
        kernel32.GlobalUnlock(handle)
        assert user32.SetClipboardData(lineedit._CF_DIB, handle)
    finally:
        user32.CloseClipboard()

    png = lineedit.read_clipboard_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(png)).size == (90, 30)


# ---------------------------------------------------------------------------
# Not only pictures: a file, a folder, an archive, a recording
# ---------------------------------------------------------------------------


def test_kind_is_named_from_the_extension():
    assert lineedit.kind_of("a.PNG") == "image"
    assert lineedit.kind_of("set.wav") == "audio"
    assert lineedit.kind_of("take.mp4") == "video"
    assert lineedit.kind_of("x.tar.gz") == "archive"
    assert lineedit.kind_of("paper.pdf") == "document"
    assert lineedit.kind_of("srv.log") == "text"
    # A real answer, not a failure: an unknown file is opaque, and saying so is
    # what stops the model inventing a format from the name.
    assert lineedit.kind_of("mystery.bin") == "other"


def test_a_file_is_a_bracketed_path_with_its_kind_and_size(tmp_path):
    target = tmp_path / "report.pdf"
    target.write_bytes(b"%PDF" + b"0" * 2048)
    line = lineedit.describe_path(str(target))
    assert line.startswith("[Attached: ")
    assert str(target) in line
    assert "document" in line and "KB" in line


def test_a_folder_says_how_many_entries_it_holds(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "one" / "a.txt").write_text("a")
    (tmp_path / "one" / "b.txt").write_text("b")
    line = lineedit.describe_path(str(tmp_path / "one"))
    assert "directory" in line and "2 entries" in line
    # ...and its contents are not spilled onto the line. A pasted folder is one
    # path; walking it is a tool call the model can make if it needs to.
    assert "a.txt" not in line


def test_a_picture_copied_as_a_file_is_attached_as_a_picture(tmp_path):
    target = tmp_path / "shot.png"
    target.write_bytes(_png("shot"))
    refs = lineedit.collapse_paths([str(target)], directory=tmp_path)
    assert len(refs) == 1
    assert refs[0].startswith(lineedit.IMAGE_MARK)


def test_a_mixed_clipboard_pastes_everything_it_holds(tmp_path, monkeypatch):
    folder = tmp_path / "out"
    folder.mkdir()
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "bundle.zip").write_bytes(b"PK\x03\x04")
    monkeypatch.setattr(lineedit, "read_clipboard_files",
                        lambda: [str(tmp_path / "notes.txt"), str(tmp_path / "bundle.zip"), str(folder)])
    monkeypatch.setattr(lineedit, "read_clipboard", lambda: "")
    monkeypatch.setattr(lineedit, "read_clipboard_png", lambda: b"")
    editor = lineedit.LineEditor(stream=io.StringIO(), paste_to=tmp_path)
    assert editor._paste() is True
    lines = "".join(editor._buf).split("\n")
    assert len(lines) == 3
    assert "notes.txt (text," in lines[0]
    assert "bundle.zip (archive," in lines[1]
    assert "(directory," in lines[2]


def test_the_paste_leaves_a_receipt_on_the_way_in(tmp_path, monkeypatch):
    monkeypatch.setattr(lineedit, "read_clipboard_png", lambda: _png())
    monkeypatch.setattr(lineedit, "read_clipboard", lambda: "")
    editor = lineedit.LineEditor(stream=io.StringIO(), paste_to=tmp_path)
    assert editor.take_notice() == ""
    editor._paste()
    assert editor.take_notice().startswith("\u2713")
    # Once: a receipt repeated on the next line would be a receipt for nothing.
    assert editor.take_notice() == ""


def test_the_channel_prints_the_receipt_when_the_line_is_sent(monkeypatch):
    """The person pasting has no other way to know it took."""
    printed: list[str] = []
    steering = Steering(stream=io.StringIO(), printer=printed.append)
    monkeypatch.setattr(steering.editor, "take_notice", lambda: "\u2713 attached: shot.png")
    steering.submit("what is this?")
    assert printed and printed[0].startswith("\u2713")


def test_a_blank_line_still_prints_the_receipt(monkeypatch):
    """Enter on a line that holds nothing but an attachment is the common case:
    the paste *is* the message."""
    printed: list[str] = []
    steering = Steering(stream=io.StringIO(), printer=printed.append)
    monkeypatch.setattr(steering.editor, "take_notice", lambda: "\u2713 attached: shot.png")
    steering.submit("")
    assert printed == ["\u2713 attached: shot.png"]


def test_an_empty_clipboard_still_pastes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(lineedit, "read_clipboard", lambda: "")
    monkeypatch.setattr(lineedit, "read_clipboard_png", lambda: b"")
    monkeypatch.setattr(lineedit, "read_clipboard_files", lambda: [])
    editor = lineedit.LineEditor(stream=io.StringIO(), paste_to=tmp_path)
    assert editor._paste() is False
    assert editor._buf == [] and editor.notice == ""
