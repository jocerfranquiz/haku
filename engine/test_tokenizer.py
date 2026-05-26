from __future__ import annotations

from engine.tokenizer import count, decode, encode


def test_roundtrip_ascii() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    ids = encode(text)
    assert len(ids) > 0
    assert decode(ids) == text


def test_roundtrip_spanish() -> None:
    text = "El niño comió una manzana en el café."
    ids = encode(text)
    assert len(ids) > 0
    assert decode(ids) == text


def test_count_matches_encode_length() -> None:
    text = "hello world"
    assert count(text) == len(encode(text))


def test_empty_string() -> None:
    assert decode([]) == ""
