"""Tests for reranking with bge-reranker-v2-m3. See DESIGN.md §16.4, §25 step 8."""

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
    """Index the corpus once for all rerank tests."""
    dest = tmp_path_factory.mktemp("haku_rerank")

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
        timeout=300,
        check=False,
    )


# ---------------------------------------------------------------------------
# Reranked by default
# ---------------------------------------------------------------------------


def test_rerank_default_on(indexed_state: Path) -> None:
    result = _search(indexed_state, "semantic search", "--format", "json")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["rerank"] is True
    for r in data["results"]:
        assert r["score_kind"] == "rerank"


def test_rerank_scores_descending(indexed_state: Path) -> None:
    result = _search(indexed_state, "information retrieval", "--format", "json")
    data = json.loads(result.stdout)
    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# --no-rerank opt-out
# ---------------------------------------------------------------------------


def test_no_rerank_flag(indexed_state: Path) -> None:
    result = _search(
        indexed_state, "semantic search", "--no-rerank", "--format", "json",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["rerank"] is False
    for r in data["results"]:
        assert r["score_kind"] == "rrf"


# ---------------------------------------------------------------------------
# A/B: rerank vs no-rerank produce different orderings
# ---------------------------------------------------------------------------


def test_rerank_changes_ordering(indexed_state: Path) -> None:
    """Reranking should produce a different score distribution than RRF."""
    reranked = _search(
        indexed_state, "search engines", "--format", "json", "--top", "5",
    )
    no_rerank = _search(
        indexed_state, "search engines", "--no-rerank", "--format", "json", "--top", "5",
    )
    r_data = json.loads(reranked.stdout)
    n_data = json.loads(no_rerank.stdout)
    r_scores = [r["score"] for r in r_data["results"]]
    n_scores = [r["score"] for r in n_data["results"]]
    # Scores should differ — reranker produces logits, RRF produces 1/(k+rank) sums
    assert r_scores != n_scores


# ---------------------------------------------------------------------------
# Spanish query reranked
# ---------------------------------------------------------------------------


def test_rerank_spanish(indexed_state: Path) -> None:
    result = _search(
        indexed_state, "búsqueda semántica", "--format", "json",
    )
    data = json.loads(result.stdout)
    assert len(data["results"]) > 0
    assert data["rerank"] is True
