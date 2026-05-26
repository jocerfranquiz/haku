"""~95-line markdown splitter. See DESIGN.md §13."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from engine.tokenizer import decode, encode


@dataclass
class Chunk:
    chunk_idx: int  # 0-based, assigned by the chunker
    text: str
    token_count: int
    source_path: str  # threaded through from the caller
    start_offset: int  # char offset into source_path; non-overlapped core
    end_offset: int


def _split_sections(md: str) -> list[tuple[int, str]]:
    """Split on H2 (^## ) boundaries. Returns [(abs_char_offset, section_text), ...]."""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start, cur = 0, 0
    for line in md.splitlines(keepends=True):
        if line.startswith("## ") and buf:
            out.append((buf_start, "".join(buf)))
            buf, buf_start = [], cur
        buf.append(line)
        cur += len(line)
    if buf:
        out.append((buf_start, "".join(buf)))
    return out


def _split_paragraphs(section: str, base_offset: int) -> list[tuple[int, str]]:
    """Blank-line-separated paragraphs with accurate offsets. Re-scans with find()
    so multiple consecutive blank lines don't drift offsets."""
    out: list[tuple[int, str]] = []
    sep = "\n\n"
    i, n = 0, len(section)
    while i < n:
        while i < n and section.startswith("\n", i):
            i += 1
        if i >= n:
            break
        j = section.find(sep, i)
        end = n if j == -1 else j
        para = section[i:end]
        if para.strip():
            out.append((base_offset + i, para))
        i = end + len(sep) if j != -1 else n
    return out


def chunk_markdown(
    md: str,
    source_path: str,
    max_tokens: int = 512,
    overlap: int = 64,
) -> Iterable[Chunk]:
    """Split -> pack to max_tokens -> apply overlap across boundaries."""
    paras: list[tuple[int, str]] = []
    for off, section in _split_sections(md):
        paras.extend(_split_paragraphs(section, off))

    # Greedy-pack paragraphs up to max_tokens.
    cores: list[tuple[int, int, str, int]] = []  # (start, end, text, tokens)
    cur_start: int | None = None
    cur_end, cur_tokens = 0, 0
    cur_text = ""
    for off, para in paras:
        n_tok = len(encode(para))
        if cur_start is None:
            cur_start = off
        if cur_tokens + n_tok <= max_tokens:
            cur_text = (cur_text + "\n\n" + para) if cur_text else para
            cur_tokens += n_tok
            cur_end = off + len(para)
        else:
            if cur_text:
                cores.append((cur_start, cur_end, cur_text, cur_tokens))
            cur_start, cur_end, cur_text, cur_tokens = off, off + len(para), para, n_tok
    if cur_text and cur_start is not None:
        cores.append((cur_start, cur_end, cur_text, cur_tokens))

    # Emit chunks, applying overlap as a tokenized prefix from the previous core.
    for idx, (s, e, text, tokens) in enumerate(cores):
        if idx == 0 or overlap <= 0:
            yield Chunk(idx, text, tokens, source_path, s, e)
            continue
        prev_ids = encode(cores[idx - 1][2])
        tail_ids = prev_ids[-overlap:]
        tail_text = decode(tail_ids)
        merged_text = tail_text + "\n\n" + text
        yield Chunk(
            chunk_idx=idx,
            text=merged_text,
            token_count=len(tail_ids) + tokens,
            source_path=source_path,
            start_offset=s,
            end_offset=e,
        )
