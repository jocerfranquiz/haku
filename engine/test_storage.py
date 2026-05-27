from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from engine.chunk import Chunk
from engine.storage import (
    EXPECTED_SCHEMA_VERSION,
    bootstrap_schema,
    check_schema_version,
    insert_chunk_with_embedding,
    insert_file,
    open_db,
    per_file_transaction,
)

EMBEDDING_DIM = 1024


def _tmp_db() -> tuple[Path, sqlite3.Connection]:
    """Create a temp DB with schema bootstrapped."""
    path = Path(tempfile.mktemp(suffix=".db"))
    conn = open_db(path)
    bootstrap_schema(conn)
    return path, conn


# ---------------------------------------------------------------------------
# DB creation and schema bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_creates_tables() -> None:
    path, conn = _tmp_db()
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        assert "schema_version" in tables
        assert "files" in tables
        assert "chunks" in tables
        assert "index_runs" in tables
    finally:
        conn.close()
        path.unlink(missing_ok=True)


def test_bootstrap_sets_schema_version() -> None:
    path, conn = _tmp_db()
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row is not None
        assert row[0] == EXPECTED_SCHEMA_VERSION
    finally:
        conn.close()
        path.unlink(missing_ok=True)


def test_wal_mode_enabled() -> None:
    path, conn = _tmp_db()
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()
        assert mode is not None
        assert mode[0] == "wal"
    finally:
        conn.close()
        path.unlink(missing_ok=True)


def test_sqlite_vec_loaded() -> None:
    path, conn = _tmp_db()
    try:
        row = conn.execute("SELECT vec_version()").fetchone()
        assert row is not None
        assert row[0]  # non-empty version string
    finally:
        conn.close()
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Schema version mismatch
# ---------------------------------------------------------------------------


def test_check_schema_version_passes_on_match() -> None:
    path, conn = _tmp_db()
    try:
        check_schema_version(conn)  # should not raise
    finally:
        conn.close()
        path.unlink(missing_ok=True)


def test_check_schema_version_refuses_on_mismatch() -> None:
    path, conn = _tmp_db()
    try:
        conn.execute("UPDATE schema_version SET version = 999")
        conn.commit()
        with pytest.raises(SystemExit, match=r"database schema version 999 found"):
            check_schema_version(conn)
    finally:
        conn.close()
        path.unlink(missing_ok=True)


def test_check_schema_version_refuses_on_empty() -> None:
    path, conn = _tmp_db()
    try:
        conn.execute("DELETE FROM schema_version")
        conn.commit()
        with pytest.raises(SystemExit, match=r"no schema_version found"):
            check_schema_version(conn)
    finally:
        conn.close()
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Per-file transaction + insert helpers
# ---------------------------------------------------------------------------


def _dummy_chunk(idx: int = 0) -> Chunk:
    return Chunk(
        chunk_idx=idx,
        text="Hello world.",
        token_count=3,
        source_path="/tmp/test.md",
        start_offset=0,
        end_offset=12,
    )


def _zero_embedding() -> list[float]:
    return [0.0] * EMBEDDING_DIM


def test_insert_chunk_with_embedding_atomic() -> None:
    path, conn = _tmp_db()
    try:
        with per_file_transaction(conn):
            file_id = insert_file(
                conn,
                path="/tmp/test.md",
                source_path="/tmp/test.md",
                file_hash="abc123",
                mtime=1000.0,
                size=100,
            )
            chunk_id = insert_chunk_with_embedding(conn, file_id, _dummy_chunk(), _zero_embedding())

        # verify both tables have the row
        chunk_row = conn.execute("SELECT id, text FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        assert chunk_row is not None
        assert chunk_row[1] == "Hello world."

        vec_row = conn.execute(
            "SELECT rowid FROM vec_chunks WHERE rowid = ?", (chunk_id,),
        ).fetchone()
        assert vec_row is not None
        assert vec_row[0] == chunk_id
    finally:
        conn.close()
        path.unlink(missing_ok=True)


def test_transaction_rollback_on_error() -> None:
    path, conn = _tmp_db()
    try:
        with pytest.raises(ValueError, match="intentional"), per_file_transaction(conn):
                insert_file(
                    conn,
                    path="/tmp/test.md",
                    source_path="/tmp/test.md",
                    file_hash="abc123",
                    mtime=1000.0,
                    size=100,
                )
                raise ValueError("intentional")

        # file row should not exist after rollback
        row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
        assert row is not None
        assert row[0] == 0
    finally:
        conn.close()
        path.unlink(missing_ok=True)


def test_fts_trigger_insert_and_delete() -> None:
    """Smoke test: FTS5 triggers keep fts_chunks in sync. See §19."""
    path, conn = _tmp_db()
    try:
        with per_file_transaction(conn):
            file_id = insert_file(
                conn,
                path="/tmp/test.md",
                source_path="/tmp/test.md",
                file_hash="abc123",
                mtime=1000.0,
                size=100,
            )
            insert_chunk_with_embedding(conn, file_id, _dummy_chunk(), _zero_embedding())

        # FTS should find it
        fts_hit = conn.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH 'hello'",
        ).fetchone()
        assert fts_hit is not None

        # Delete the file (CASCADE deletes chunks, trigger cleans FTS)
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()

        fts_after = conn.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH 'hello'",
        ).fetchone()
        assert fts_after is None
    finally:
        conn.close()
        path.unlink(missing_ok=True)


def test_multiple_chunks_per_file() -> None:
    path, conn = _tmp_db()
    try:
        with per_file_transaction(conn):
            file_id = insert_file(
                conn,
                path="/tmp/multi.md",
                source_path="/tmp/multi.md",
                file_hash="def456",
                mtime=2000.0,
                size=200,
            )
            for i in range(3):
                insert_chunk_with_embedding(conn, file_id, _dummy_chunk(idx=i), _zero_embedding())

        count = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_id = ?", (file_id,)).fetchone()
        assert count is not None
        assert count[0] == 3
    finally:
        conn.close()
        path.unlink(missing_ok=True)
