"""Tests for `haku search` — vector-only retrieval. See DESIGN.md §16, §25 step 6."""

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
    """Index the corpus once for all search tests in this module."""
    dest = tmp_path_factory.mktemp("haku")

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
# Basic search
# ---------------------------------------------------------------------------


def test_search_returns_results(indexed_state: Path) -> None:
    result = _search(indexed_state, "semantic search")
    assert result.returncode == 0
    assert "[" in result.stdout  # score bracket


def test_search_top_limits_results(indexed_state: Path) -> None:
    result = _search(indexed_state, "information retrieval", "--top", "2")
    assert result.returncode == 0
    assert "1." in result.stdout
    assert "2." in result.stdout
    assert "3." not in result.stdout


def test_search_default_top_is_five(indexed_state: Path) -> None:
    result = _search(indexed_state, "search engines")
    assert result.returncode == 0
    assert "1." in result.stdout
    # Should have at most 5 results
    assert "6." not in result.stdout


# ---------------------------------------------------------------------------
# JSON output (§17.2)
# ---------------------------------------------------------------------------


def test_search_json_format(indexed_state: Path) -> None:
    result = _search(indexed_state, "semantic search", "--format", "json")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["schema_version"] == 1
    assert data["query"] == "semantic search"
    assert isinstance(data["rerank"], bool)
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0
    first = data["results"][0]
    assert "rank" in first
    assert "score" in first
    assert "path" in first
    assert "snippet" in first
    assert "chunk_idx" in first
    assert "start_offset" in first
    assert "end_offset" in first


def test_search_json_scores_descending(indexed_state: Path) -> None:
    result = _search(indexed_state, "information retrieval", "--format", "json")
    data = json.loads(result.stdout)
    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Spanish query (bilingual quality)
# ---------------------------------------------------------------------------


def test_search_spanish_query(indexed_state: Path) -> None:
    result = _search(
        indexed_state, "búsqueda semántica", "--format", "json",
    )
    data = json.loads(result.stdout)
    assert len(data["results"]) > 0
    paths = [r["path"] for r in data["results"]]
    assert any("notas_es" in p for p in paths)


# ---------------------------------------------------------------------------
# --files scoping (§16.3)
# ---------------------------------------------------------------------------


def test_search_scoped_to_file(indexed_state: Path) -> None:
    corpus = indexed_state / "corpus"
    md_path = str(corpus / "notes.md")
    result = _search(
        indexed_state, "retrieval", "--files", md_path, "--format", "json",
    )
    data = json.loads(result.stdout)
    for r in data["results"]:
        assert r["path"] == md_path


def test_search_unindexed_scope_fails(indexed_state: Path) -> None:
    result = _search(indexed_state, "test", "--files", "/nonexistent/dir")
    assert result.returncode != 0
    assert "no indexed files" in result.stderr


# ---------------------------------------------------------------------------
# Output file (--output)
# ---------------------------------------------------------------------------


def test_search_output_to_file(indexed_state: Path) -> None:
    out_path = str(indexed_state / "results.json")
    result = _search(
        indexed_state, "search", "--format", "json", "--output", out_path,
    )
    assert result.returncode == 0
    data = json.loads(Path(out_path).read_text())
    assert len(data["results"]) > 0
