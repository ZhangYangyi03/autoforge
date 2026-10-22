"""The index's lexical leg: a tokenizer fix has to reach documents already stored.

The bug being guarded, measured 2026-09-17: `mcp_x_pdf_merge` was indexed, a
query for `pdf merge` returned nothing, and the document was sitting right there.
`_WORD` treats an underscore-joined identifier as ONE token, so the name
contributed the single opaque term `mcp_x_pdf_merge` and shared no term with any
query about pdf. The dense leg hid this on the machine it was built on; on any
shelf reached without the embedder -- another process, a rebuilt index, a model
that is not the one that built it -- lexical is all there is, and it answered 0.
"""
from __future__ import annotations

import pytest

from toolmarket import vectorstore as vs


def test_identifier_is_also_indexed_as_its_parts():
    toks = vs.tokenize("mcp_x_pdf_merge")
    assert "pdf" in toks and "merge" in toks
    # The stem stays too, so typing the whole name still matches it exactly.
    assert "mcp_x_pdf_merge" in toks


def test_plain_words_are_untouched():
    assert vs.tokenize("pdf merge") == ["pdf", "merge"]


def test_frequency_is_not_double_counted_for_plain_words():
    """The parts are only added for a joined word, not for every word."""
    toks = vs.tokenize("merge merge")
    assert toks.count("merge") == 2


@pytest.fixture()
def store(tmp_path):
    st = vs.Store(str(tmp_path / "i.sqlite"))
    st.create_collection("tools")
    return st


def test_a_query_for_a_word_inside_the_name_now_matches(store):
    """The regression, at the level it was measured: bm25_search, not /search."""
    store.index_text("tools", "tool:mcp_x_pdf_merge",
                     vs.build_resource_text({"name": "mcp_x_pdf_merge",
                                             "description": "does a thing"}))
    assert store.bm25_search("tools", "pdf", k=5), \
        "a word inside the identifier must retrieve it"
    assert store.bm25_search("tools", "pdf merge", k=5)
    assert store.bm25_search("tools", "mcp_x_pdf_merge", k=5)


def test_an_unrelated_query_still_misses(store):
    store.index_text("tools", "tool:mcp_x_pdf_merge",
                     vs.build_resource_text({"name": "mcp_x_pdf_merge",
                                             "description": "does a thing"}))
    assert store.bm25_search("tools", "transcode video", k=5) == []


def test_a_tokenizer_change_forces_a_reindex_of_stored_documents(store):
    """The trap: the row holds term frequencies, not text, so an old row stays
    silently wrong after a tokenizer fix unless the content hash moves with it."""
    store.index_text("tools", "tool:x", "mcp_x_pdf_merge")
    chash = vs.hash_text("mcp_x_pdf_merge")
    assert store.needs_reindex("tools", "tool:x", chash) is False
    assert store.index_text("tools", "tool:x", "mcp_x_pdf_merge") is False, \
        "unchanged text is not rewritten"

    # What an unversioned hash would have produced for the same text.
    import hashlib
    unversioned = hashlib.sha256(b"mcp_x_pdf_merge").hexdigest()[:16]
    assert unversioned != chash, "the version must be part of the hash"
    assert store.needs_reindex("tools", "tool:x", unversioned) is True

    # And the version is the only reason: same text, same version, no rewrite.
    assert vs.hash_text("mcp_x_pdf_merge") == chash


def test_sync_rewrites_documents_written_by_an_older_tokenizer(store):
    """End to end: a stale row is picked up by the ordinary sync path."""
    store.index_text("tools", "tool:x", "mcp_x_pdf_merge",
                     chash="0" * 16)  # what an older version would have stored
    rec = {"id": "tool:x", "name": "mcp_x_pdf_merge", "description": "does a thing"}
    report = store.sync_from_records("tools", [rec])
    assert report["written"] == 1
    assert store.bm25_search("tools", "pdf", k=5)
