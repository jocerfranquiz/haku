"""Tests for hybrid search (vector + FTS5 + RRF). See DESIGN.md §16, §25 step 7."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HAKU_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = HAKU_ROOT / "engine" / "fixtures" / "corpus"
HAKU_BIN = HAKU_ROOT / "haku"


@pytest.fixture(scope="module")
def indexed_state(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Index the corpus once for all hybrid search tests."""
    dest = tmp_path_factory.mktemp("haku_hybrid")

    (dest / "engine").symlink_to(HAKU_ROOT / "engine")
    (dest / "models").symlink_to(HAKU_ROOT / "models")
    (dest / ".venv").symlink_to(HAKU_ROOT / ".venv")
    shutil.copy2(HAKU_BIN, dest / "haku")
    shutil.copytree(CORPUS_DIR, dest / "corpus")

    env = {**os.environ, "HAKU_HOME": str(dest)}
    subprocess.run(
        [str(dest / "haku"), "index", "--files", str(dest / "corpus"), "--quiet"],
        env=env,
        timeout=300,
        check=True,
    )
    return dest


def _search(
    haku_home: Path, query: str, *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HAKU_HOME": str(haku_home)}
    return subprocess.run(
        [str(haku_home / "haku"), "search", query, *extra_args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


# ---------------------------------------------------------------------------
# RRF fusion basics
# ---------------------------------------------------------------------------


def test_hybrid_returns_results(indexed_state: Path) -> None:
    result = _search(indexed_state, "semantic search")
    assert result.returncode == 0
    assert "1." in result.stdout


def test_hybrid_json_has_rrf_score_kind_when_no_rerank(indexed_state: Path) -> None:
    result = _search(indexed_state, "search engines", "--no-rerank", "--format", "json")
    data = json.loads(result.stdout)
    assert len(data["results"]) > 0
    for r in data["results"]:
        assert r["score_kind"] == "rrf"


def test_hybrid_scores_descending(indexed_state: Path) -> None:
    result = _search(indexed_state, "information retrieval", "--format", "json")
    data = json.loads(result.stdout)
    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# FTS5 lexical matching
# ---------------------------------------------------------------------------


def test_fts_finds_exact_keyword(indexed_state: Path) -> None:
    """A rare keyword present in only one doc should rank that doc highly."""
    result = _search(indexed_state, "PageRank", "--format", "json")
    data = json.loads(result.stdout)
    assert len(data["results"]) > 0


def test_fts_spanish_diacritics_folded(indexed_state: Path) -> None:
    """FTS5 with unicode61 remove_diacritics 2 should match café/cafe. See §16.1."""
    result = _search(indexed_state, "cafe", "--format", "json")
    data = json.loads(result.stdout)
    paths = [r["path"] for r in data["results"]]
    assert any("notas_es" in p for p in paths)


# ---------------------------------------------------------------------------
# Scoping applies to both retrieval paths (§16.3)
# ---------------------------------------------------------------------------


def test_scoped_search_hybrid(indexed_state: Path) -> None:
    md_path = str(indexed_state / "corpus" / "notes.md")
    result = _search(
        indexed_state, "retrieval", "--files", md_path, "--format", "json",
    )
    data = json.loads(result.stdout)
    for r in data["results"]:
        assert r["path"] == md_path


# ---------------------------------------------------------------------------
# Top limit
# ---------------------------------------------------------------------------


def test_top_limits_hybrid(indexed_state: Path) -> None:
    result = _search(indexed_state, "search", "--top", "2", "--format", "json")
    data = json.loads(result.stdout)
    assert len(data["results"]) <= 2
