from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

# see §14 — anchored at __file__, not HAKU_HOME
HAKU_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _tk() -> Tokenizer:
    path = HAKU_ROOT / "models" / "qwen3-embedding-0.6b" / "tokenizer.json"
    return Tokenizer.from_file(str(path))


def encode(text: str) -> list[int]:
    return _tk().encode(text).ids


def decode(ids: list[int]) -> str:
    return _tk().decode(ids)


def count(text: str) -> int:
    return len(encode(text))


def get_tokenizer() -> Tokenizer:
    """Public accessor for embed.py batch encoding. See §14."""
    return _tk()
