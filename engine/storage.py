"""SQLite schema bootstrap, version check, and transaction helpers. See DESIGN.md §19."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import sqlite_vec  # type: ignore[import-untyped]

from engine.chunk import Chunk

if TYPE_CHECKING:
    from collections.abc import Iterator

EXPECTED_SCHEMA_VERSION = 1

HAKU_ROOT = Path(__file__).resolve().parent.parent

# see §19 — full schema DDL
_SCHEMA_DDL = """\
CREATE TABLE schema_version (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE files (
  id          INTEGER PRIMARY KEY,
  path        TEXT NOT NULL UNIQUE,
  source_path TEXT NOT NULL,
  hash        TEXT NOT NULL,
  mtime       REAL NOT NULL,
  size        INTEGER NOT NULL,
  indexed_at  TEXT NOT NULL
);
CREATE INDEX idx_files_hash ON files(hash);

CREATE TABLE chunks (
  id           INTEGER PRIMARY KEY,
  file_id      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_idx    INTEGER NOT NULL,
  text         TEXT NOT NULL,
  token_count  INTEGER NOT NULL,
  start_offset INTEGER NOT NULL,
  end_offset   INTEGER NOT NULL,
  UNIQUE(file_id, chunk_idx)
);
CREATE INDEX idx_chunks_file ON chunks(file_id);

CREATE VIRTUAL TABLE vec_chunks USING vec0(
  embedding float[1024]
);

CREATE VIRTUAL TABLE fts_chunks USING fts5(
  text,
  content='chunks',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO fts_chunks(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES('delete', old.id, old.text);
  INSERT INTO fts_chunks(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE index_runs (
  run_id         INTEGER PRIMARY KEY,
  started_at     TEXT NOT NULL,
  finished_at    TEXT,
  indexed_count  INTEGER NOT NULL DEFAULT 0,
  skipped_count  INTEGER NOT NULL DEFAULT 0,
  failed_count   INTEGER NOT NULL DEFAULT 0
);
"""


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension. See §7."""
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:
        msg = (
            "haku: sqlite-vec failed to load: the system Python's sqlite3 module was built\n"
            "      without --enable-loadable-sqlite-extensions. Reinstall Python (pyenv,\n"
            "      uv, or a distribution that ships it enabled) and recreate /haku/.venv."
        )
        raise SystemExit(msg) from exc


def open_db(db_path: Path | str) -> sqlite3.Connection:
    """Open a connection with WAL mode and sqlite-vec loaded."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _load_sqlite_vec(conn)
    return conn


def bootstrap_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, indexes, triggers, and insert schema_version. See §19."""
    conn.executescript(_SCHEMA_DDL)
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (EXPECTED_SCHEMA_VERSION, now),
    )
    conn.commit()


def check_schema_version(conn: sqlite3.Connection) -> None:
    """Refuse to run on schema version mismatch. See §19."""
    row = conn.execute(
        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1",
    ).fetchone()
    if row is None:
        msg = "haku: no schema_version found in database. Re-run `haku init`."
        raise SystemExit(msg)
    found = row[0]
    if found != EXPECTED_SCHEMA_VERSION:
        # see §19 — exact error message
        msg = (
            f"haku: database schema version {found} found, "
            f"this haku expects version {EXPECTED_SCHEMA_VERSION}.\n"
            "      back up /haku/database.db if needed, then delete it "
            "and re-run `haku index`."
        )
        raise SystemExit(msg)


def insert_file(  # noqa: PLR0913
    conn: sqlite3.Connection,
    *,
    path: str,
    source_path: str,
    file_hash: str,
    mtime: float,
    size: int,
) -> int:
    """Insert a file record and return its id."""
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        "INSERT INTO files (path, source_path, hash, mtime, size, indexed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (path, source_path, file_hash, mtime, size, now),
    )
    return cur.lastrowid  # type: ignore[return-value]


def insert_chunk_with_embedding(
    conn: sqlite3.Connection,
    file_id: int,
    chunk: Chunk,
    embedding: list[float],
) -> int:
    """Insert a chunk row and its vec_chunks embedding atomically. See §19 contract:
    this is the ONLY supported insertion path for vec_chunks."""
    cur = conn.execute(
        "INSERT INTO chunks (file_id, chunk_idx, text, token_count, start_offset, end_offset) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (file_id, chunk.chunk_idx, chunk.text, chunk.token_count,
         chunk.start_offset, chunk.end_offset),
    )
    chunk_id: int = cur.lastrowid  # type: ignore[assignment]
    conn.execute(
        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
        (chunk_id, sqlite_vec.serialize_float32(embedding)),
    )
    return chunk_id


@contextmanager
def per_file_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Context manager for a per-file atomic transaction. See §12.6."""
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
