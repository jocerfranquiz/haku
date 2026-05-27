"""Tests for CLI commands: init, status, purge, --version, --quiet. Step 9."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HAKU_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = HAKU_ROOT / "engine" / "fixtures" / "corpus"
HAKU_BIN = HAKU_ROOT / "haku"


def _make_haku_home(tmp_path: Path) -> Path:
    """Create a minimal haku home dir with symlinks."""
    dest = tmp_path / "haku"
    dest.mkdir()
    (dest / "engine").symlink_to(HAKU_ROOT / "engine")
    (dest / "models").symlink_to(HAKU_ROOT / "models")
    (dest / ".venv").symlink_to(HAKU_ROOT / ".venv")
    shutil.copy2(HAKU_BIN, dest / "haku")
    return dest


def _run(
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
# haku init (§6)
# ---------------------------------------------------------------------------


def test_init_creates_dirs(tmp_path: Path) -> None:
    home = _make_haku_home(tmp_path)
    _run(home, "init", "--quiet")
    assert (home / "markdowns").exists()
    assert (home / "logs").exists()
    assert (home / "database.db").exists()


def test_init_idempotent(tmp_path: Path) -> None:
    home = _make_haku_home(tmp_path)
    _run(home, "init", "--quiet")
    _run(home, "init", "--quiet")  # second run should not fail


# ---------------------------------------------------------------------------
# haku status (§15)
# ---------------------------------------------------------------------------


def test_status_shows_model_status(tmp_path: Path) -> None:
    home = _make_haku_home(tmp_path)
    _run(home, "init", "--quiet")
    result = _run(home, "status")
    assert "embedder:" in result.stdout
    assert "ok" in result.stdout
    assert "reranker:" in result.stdout


def test_status_shows_zero_files_before_index(tmp_path: Path) -> None:
    home = _make_haku_home(tmp_path)
    _run(home, "init", "--quiet")
    result = _run(home, "status")
    assert "indexed files:           0" in result.stdout


def test_status_fails_without_db(tmp_path: Path) -> None:
    home = _make_haku_home(tmp_path)
    result = _run(home, "status", expect_fail=True)
    assert result.returncode != 0
    assert "no database" in result.stderr


# ---------------------------------------------------------------------------
# haku purge (§18)
# ---------------------------------------------------------------------------


@pytest.fixture()
def indexed_home(tmp_path: Path) -> Path:
    """A haku home with corpus indexed."""
    home = _make_haku_home(tmp_path)
    shutil.copytree(CORPUS_DIR, home / "corpus")
    _run(home, "init", "--quiet")
    _run(home, "index", "--files", str(home / "corpus"), "--quiet")
    return home


def test_purge_nothing_to_purge(indexed_home: Path) -> None:
    result = _run(indexed_home, "purge")
    assert "nothing to purge" in result.stderr


def test_purge_removes_deleted_files(indexed_home: Path) -> None:
    (indexed_home / "corpus" / "readme.txt").unlink()
    result = _run(indexed_home, "purge")
    assert "purged 1 file" in result.stderr


# ---------------------------------------------------------------------------
# haku --version (§11)
# ---------------------------------------------------------------------------


def test_version_output(tmp_path: Path) -> None:
    home = _make_haku_home(tmp_path)
    _run(home, "init", "--quiet")
    result = _run(home, "--version")
    assert "haku 0.1.0" in result.stdout
    assert "embedder:" in result.stdout
    assert "reranker:" in result.stdout
    assert "licenses:" in result.stdout


def test_version_full(tmp_path: Path) -> None:
    home = _make_haku_home(tmp_path)
    _run(home, "init", "--quiet")
    result = _run(home, "--version", "--full")
    assert "haku 0.1.0" in result.stdout
    # Full SHA should be longer than 8 chars
    for line in result.stdout.splitlines():
        if "revision:" in line:
            rev = line.split(":")[-1].strip()
            assert len(rev) == 40


# ---------------------------------------------------------------------------
# --quiet (§9)
# ---------------------------------------------------------------------------


def test_quiet_suppresses_init_output(tmp_path: Path) -> None:
    home = _make_haku_home(tmp_path)
    result = _run(home, "init", "--quiet")
    # Only stderr from pip install should appear, not the "haku initialized" message
    assert "haku initialized" not in result.stderr


def test_quiet_suppresses_status(tmp_path: Path) -> None:
    home = _make_haku_home(tmp_path)
    _run(home, "init", "--quiet")
    result = _run(home, "status", "--quiet")
    assert result.stdout.strip() == ""
