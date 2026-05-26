from __future__ import annotations

from pathlib import Path

from engine.chunk import Chunk, chunk_markdown

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


# ---------------------------------------------------------------------------
# Helper: common structural assertions reused across both language fixtures
# ---------------------------------------------------------------------------


def _assert_chunk_invariants(chunks: list[Chunk], source_path: str) -> None:
    assert len(chunks) > 0
    for i, c in enumerate(chunks):
        assert c.chunk_idx == i
        assert c.source_path == source_path
        assert c.start_offset < c.end_offset
        assert c.token_count > 0
        assert len(c.text) > 0
    # cores must be non-overlapping
    for i in range(1, len(chunks)):
        assert chunks[i].start_offset >= chunks[i - 1].end_offset


# ---------------------------------------------------------------------------
# English golden-file tests
# ---------------------------------------------------------------------------


def test_en_chunk_count() -> None:
    md = _load("en_doc.md")
    chunks = list(chunk_markdown(md, "en_doc.md", max_tokens=100, overlap=20))
    assert len(chunks) == 6


def test_en_structural_invariants() -> None:
    md = _load("en_doc.md")
    chunks = list(chunk_markdown(md, "en_doc.md", max_tokens=100, overlap=20))
    _assert_chunk_invariants(chunks, "en_doc.md")


def test_en_first_chunk_has_no_overlap_prefix() -> None:
    md = _load("en_doc.md")
    chunks = list(chunk_markdown(md, "en_doc.md", max_tokens=100, overlap=20))
    assert chunks[0].text.startswith("# The History of Search Engines")


def test_en_second_chunk_has_overlap_prefix() -> None:
    """Chunks after the first carry a decoded-token prefix from the previous core."""
    md = _load("en_doc.md")
    chunks = list(chunk_markdown(md, "en_doc.md", max_tokens=100, overlap=20))
    # The second chunk's text should NOT start at its core start_offset —
    # it has an overlap prefix prepended.
    core_text = md[chunks[1].start_offset : chunks[1].end_offset]
    assert chunks[1].text.endswith(core_text)
    assert len(chunks[1].text) > len(core_text)


def test_en_no_overlap_mode() -> None:
    md = _load("en_doc.md")
    chunks = list(chunk_markdown(md, "en_doc.md", max_tokens=100, overlap=0))
    _assert_chunk_invariants(chunks, "en_doc.md")
    for c in chunks:
        core_text = md[c.start_offset : c.end_offset]
        assert c.text == core_text


# ---------------------------------------------------------------------------
# Spanish golden-file tests
# ---------------------------------------------------------------------------


def test_es_chunk_count() -> None:
    md = _load("es_doc.md")
    chunks = list(chunk_markdown(md, "es_doc.md", max_tokens=100, overlap=20))
    assert len(chunks) == 6


def test_es_structural_invariants() -> None:
    md = _load("es_doc.md")
    chunks = list(chunk_markdown(md, "es_doc.md", max_tokens=100, overlap=20))
    _assert_chunk_invariants(chunks, "es_doc.md")


def test_es_accented_characters_preserved() -> None:
    md = _load("es_doc.md")
    chunks = list(chunk_markdown(md, "es_doc.md", max_tokens=100, overlap=20))
    full_text = "\n\n".join(c.text for c in chunks)
    assert "Búsqueda" in full_text
    assert "búsqueda" in full_text
    assert "Verónica" in full_text
    assert "categorías" in full_text


def test_es_first_chunk_starts_at_h1() -> None:
    md = _load("es_doc.md")
    chunks = list(chunk_markdown(md, "es_doc.md", max_tokens=100, overlap=20))
    assert chunks[0].text.startswith("# Historia de los Motores")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_paragraph_yields_one_chunk() -> None:
    md = "Just one paragraph, no headings."
    chunks = list(chunk_markdown(md, "test.md", max_tokens=512, overlap=64))
    assert len(chunks) == 1
    assert chunks[0].text == md
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(md)


def test_h2_splits_sections() -> None:
    md = "## Section A\n\nParagraph A.\n\n## Section B\n\nParagraph B."
    chunks = list(chunk_markdown(md, "test.md", max_tokens=512, overlap=0))
    assert len(chunks) == 1  # both fit in 512 tokens
    assert "Section A" in chunks[0].text
    assert "Section B" in chunks[0].text
