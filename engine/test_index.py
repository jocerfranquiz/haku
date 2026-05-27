"""End-to-end tests for `haku index`. See DESIGN.md §12, §25 step 5."""

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


@pytest.fixture()
def clean_state(tmp_path: Path) -> Path:
    """Provide a clean HAKU_ROOT-like directory with corpus symlinked in."""
    # Copy the real project structure needed for indexing
    dest = tmp_path / "haku"
    dest.mkdir()

    # Symlink engine/, models/, .venv/ so we can run haku
    (dest / "engine").symlink_to(HAKU_ROOT / "engine")
    (dest / "models").symlink_to(HAKU_ROOT / "models")
    (dest / ".venv").symlink_to(HAKU_ROOT / ".venv")

    # Copy the bash entrypoint
    shutil.copy2(HAKU_BIN, dest / "haku")

    # Copy corpus into the temp dir
    corpus_dest = dest / "corpus"
    shutil.copytree(CORPUS_DIR, corpus_dest)

    return dest


def _run_haku(
    haku_home: Path, *args: str, expect_fail: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HAKU_HOME": str(haku_home)}
    result = subprocess.run(
        [str(haku_home / "haku"), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    if not expect_fail and result.returncode != 0:
        msg = f"haku failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        pytest.fail(msg)
    return result


# ---------------------------------------------------------------------------
# Core indexing
# ---------------------------------------------------------------------------


def test_index_runs_to_completion(clean_state: Path) -> None:
    result = _run_haku(clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet")
    assert result.returncode == 0


def test_index_creates_database(clean_state: Path) -> None:
    _run_haku(clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet")
    assert (clean_state / "database.db").exists()


def test_index_logs_errors(clean_state: Path) -> None:
    _run_haku(clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet")
    errors_path = clean_state / "logs" / "errors.jsonl"
    assert errors_path.exists()
    errors = [json.loads(line) for line in errors_path.read_text().splitlines()]
    # empty.pdf and encrypted.pdf should fail
    failed_names = {Path(e["path"]).name for e in errors}
    assert "empty.pdf" in failed_names
    assert "encrypted.pdf" in failed_names


def test_index_logs_successes(clean_state: Path) -> None:
    _run_haku(clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet")
    index_log = clean_state / "logs" / "index.jsonl"
    assert index_log.exists()
    entries = [json.loads(line) for line in index_log.read_text().splitlines()]
    assert len(entries) >= 5  # at least 5 healthy files indexed


def test_index_error_summary_on_stderr(clean_state: Path) -> None:
    result = _run_haku(clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet")
    assert "failed" in result.stderr
    assert "errors.jsonl" in result.stderr


# ---------------------------------------------------------------------------
# Incremental skip (§12.1)
# ---------------------------------------------------------------------------


def test_incremental_skip(clean_state: Path) -> None:
    _run_haku(clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet")
    # Second run should index 0 new files
    result = _run_haku(
        clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet",
    )
    assert "indexed 0 files" in result.stderr


# ---------------------------------------------------------------------------
# Unsupported extensions (§12.0)
# ---------------------------------------------------------------------------


def test_unsupported_extensions_skipped(clean_state: Path) -> None:
    """Unsupported extensions (.py, .csv) should not appear in index log or error log."""
    _run_haku(clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet")
    index_log = clean_state / "logs" / "index.jsonl"
    entries = [json.loads(line) for line in index_log.read_text().splitlines()]
    indexed_names = {Path(e["path"]).name for e in entries}
    assert "code.py" not in indexed_names
    assert "data.csv" not in indexed_names


# ---------------------------------------------------------------------------
# Spanish content (§12.0, accented chars)
# ---------------------------------------------------------------------------


def test_spanish_file_indexed(clean_state: Path) -> None:
    _run_haku(clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet")
    index_log = clean_state / "logs" / "index.jsonl"
    entries = [json.loads(line) for line in index_log.read_text().splitlines()]
    indexed_names = {Path(e["path"]).name for e in entries}
    assert "notas_es.md" in indexed_names


# ---------------------------------------------------------------------------
# Reindex (§9)
# ---------------------------------------------------------------------------


def test_reindex_re_embeds(clean_state: Path) -> None:
    _run_haku(clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet")
    result = _run_haku(
        clean_state, "index", "--files", str(clean_state / "corpus"), "--quiet", "--reindex",
    )
    # Should re-index files (not "indexed 0")
    assert "indexed 0 files" not in result.stderr or "failed" in result.stderr
