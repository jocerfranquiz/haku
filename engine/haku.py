"""haku — single dispatcher. See DESIGN.md §3."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure the project root is on sys.path so `engine.*` imports work
# when invoked directly via the bash wrapper (§8).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tqdm import tqdm  # type: ignore[import-untyped]

from engine.chunk import chunk_markdown
from engine.embed import Embedder, Reranker
from engine.storage import (
    bootstrap_schema,
    check_schema_version,
    insert_chunk_with_embedding,
    insert_file,
    open_db,
    per_file_transaction,
)

if TYPE_CHECKING:
    import sqlite3

# The bash wrapper (§8) exports HAKU_HOME; use it for DB/logs/markdowns paths.
# Falls back to __file__-based resolution per §14.
_env_home = os.environ.get("HAKU_HOME")
HAKU_ROOT = Path(_env_home) if _env_home else Path(__file__).resolve().parent.parent

# see §12.0
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".mdx", ".pdf", ".html", ".docx"}

_SUPPORTED_LIST = " ".join(sorted(SUPPORTED_EXTENSIONS))


# ---------------------------------------------------------------------------
# Logging (§22)
# ---------------------------------------------------------------------------


def _log_jsonl(path: Path, entry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _log_index(run_id: int, fpath: str, chunks: int, tokens: int, ms: int) -> None:
    _log_jsonl(
        HAKU_ROOT / "logs" / "index.jsonl",
        {
            "ts": datetime.now(UTC).isoformat(),
            "level": "info",
            "event": "file_indexed",
            "run_id": run_id,
            "path": fpath,
            "extra": {"chunks": chunks, "tokens": tokens, "ms": ms},
        },
    )


def _log_error(run_id: int, fpath: str, error: str, tb: str) -> None:
    _log_jsonl(
        HAKU_ROOT / "logs" / "errors.jsonl",
        {
            "ts": datetime.now(UTC).isoformat(),
            "level": "error",
            "event": "file_process_failed",
            "run_id": run_id,
            "path": fpath,
            "extra": {"error": error, "traceback": tb},
        },
    )


# ---------------------------------------------------------------------------
# File discovery (§12.0)
# ---------------------------------------------------------------------------


def _discover_files(paths: list[str]) -> tuple[list[Path], int]:
    """Walk --files args, return (supported files, skipped count)."""
    found: list[Path] = []
    skipped = 0
    for p in paths:
        target = Path(p).resolve()
        if target.is_file():
            if target.suffix.lower() in SUPPORTED_EXTENSIONS:
                found.append(target)
            else:
                skipped += 1
        elif target.is_dir():
            for root, _dirs, files in os.walk(target):
                for fname in files:
                    fp = Path(root) / fname
                    if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                        found.append(fp.resolve())
                    else:
                        skipped += 1
    return found, skipped


# ---------------------------------------------------------------------------
# File hashing (§12.1)
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    """sha256(path || mtime || size). See §12.1."""
    stat = path.stat()
    payload = f"{path}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Extension handlers (§12.0)
# ---------------------------------------------------------------------------


def _extract_text(path: Path) -> tuple[str, str]:
    """Extract markdown text from a file. Returns (markdown, source_path).
    source_path is the file whose char offsets the chunks refer to."""
    ext = path.suffix.lower()

    if ext in {".txt", ".md", ".markdown", ".mdx"}:
        text = path.read_text(encoding="utf-8")
        return text, str(path)

    if ext == ".pdf":
        return _extract_pdf(path)

    if ext == ".html":
        return _extract_html(path)

    if ext == ".docx":
        return _extract_docx(path)

    msg = f"unsupported extension: {ext}"
    raise ValueError(msg)


def _extract_pdf(path: Path) -> tuple[str, str]:
    """Convert PDF to markdown via PyMuPDF4LLM, cache result. See §12.2, §21.
    NOTE: PyMuPDF4LLM is AGPL v3."""
    import pymupdf4llm  # type: ignore[import-untyped]  # AGPL v3

    cache_key = hashlib.sha256(str(path).encode()).hexdigest()
    cache_path = HAKU_ROOT / "markdowns" / f"{cache_key}.md"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), str(cache_path)

    md_text: str = pymupdf4llm.to_markdown(str(path))
    cache_path.write_text(md_text, encoding="utf-8")
    return md_text, str(cache_path)


def _extract_html(path: Path) -> tuple[str, str]:
    """Strip HTML tags via selectolax. See §12.0."""
    from selectolax.parser import HTMLParser

    raw = path.read_text(encoding="utf-8")
    tree = HTMLParser(raw)
    text = tree.text(separator="\n\n") or ""
    return text, str(path)


def _extract_docx(path: Path) -> tuple[str, str]:
    """Extract text from DOCX via mammoth. See §12.0."""
    import mammoth  # type: ignore[import-untyped]

    with path.open("rb") as f:
        result = mammoth.convert_to_markdown(f)
    return result.value, str(path)


# ---------------------------------------------------------------------------
# PDF thread pool (§12.5)
# ---------------------------------------------------------------------------


def _extract_pdf_threaded(
    pdf_paths: list[Path],
) -> dict[Path, tuple[str, str] | Exception]:
    """Convert PDFs in parallel via ThreadPoolExecutor. See §12.5."""
    results: dict[Path, tuple[str, str] | Exception] = {}
    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_extract_pdf, p): p for p in pdf_paths}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                results[path] = fut.result()
            except Exception as exc:
                results[path] = exc
    return results


# ---------------------------------------------------------------------------
# Indexer lock (§12.5, §20)
# ---------------------------------------------------------------------------


def _acquire_lock() -> int:
    """Acquire exclusive non-blocking lock. Returns fd. See §20."""
    lock_path = HAKU_ROOT / ".lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        print(
            "haku: another indexer is running (lock held on /haku/.lock).",
            file=sys.stderr,
        )
        raise SystemExit(1)  # noqa: B904
    return fd


def _release_lock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


# ---------------------------------------------------------------------------
# Index command (§12)
# ---------------------------------------------------------------------------


def _cmd_index(args: argparse.Namespace) -> None:  # noqa: PLR0912, PLR0915
    quiet: bool = args.quiet
    max_tokens: int = args.chunks
    overlap: int = args.overlap
    reindex: bool = args.reindex
    file_paths: list[str] = args.files

    if not file_paths:
        print("haku: --files is required for indexing.", file=sys.stderr)
        raise SystemExit(1)

    # Discover files
    files, skipped_ext = _discover_files(file_paths)

    if not files and skipped_ext > 0:
        print(
            f"haku: indexed 0 files ({skipped_ext} skipped: unsupported extensions).\n"
            f"      supported: {_SUPPORTED_LIST}",
            file=sys.stderr,
        )
        return

    # DB setup
    db_path = HAKU_ROOT / "database.db"
    conn = open_db(db_path)
    try:
        _maybe_bootstrap(conn)
        check_schema_version(conn)
    except SystemExit:
        conn.close()
        raise

    # Lock
    lock_fd = _acquire_lock()

    try:
        # Create index run record (§19)
        now = datetime.now(UTC).isoformat()
        cur = conn.execute(
            "INSERT INTO index_runs (started_at) VALUES (?)", (now,),
        )
        conn.commit()
        run_id: int = cur.lastrowid  # type: ignore[assignment]

        # Load embedder lazily
        embedder = Embedder(args.embedder)

        indexed_count = 0
        skipped_count = 0
        failed_count = 0

        # Separate PDFs for threaded extraction (§12.5)
        pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]
        non_pdf_files = [f for f in files if f.suffix.lower() != ".pdf"]

        # Pre-extract PDFs in thread pool
        pdf_results: dict[Path, tuple[str, str] | Exception] = {}
        if pdf_files:
            pdf_results = _extract_pdf_threaded(pdf_files)

        all_files = non_pdf_files + pdf_files
        bar = tqdm(all_files, desc="indexing", disable=quiet, file=sys.stderr)

        for fpath in bar:
            t0 = time.monotonic()

            # Skip check (§12.1)
            fhash = _file_hash(fpath)
            if not reindex:
                existing = conn.execute(
                    "SELECT id FROM files WHERE hash = ?", (fhash,),
                ).fetchone()
                if existing:
                    skipped_count += 1
                    continue

            # Extract
            try:
                if fpath.suffix.lower() == ".pdf":
                    result = pdf_results.get(fpath)
                    if isinstance(result, Exception):
                        raise result
                    assert result is not None
                    md_text, source_path = result
                else:
                    md_text, source_path = _extract_text(fpath)
            except Exception as exc:
                failed_count += 1
                _log_error(run_id, str(fpath), str(exc), traceback.format_exc())
                continue

            # Chunk
            chunks = list(
                chunk_markdown(md_text, source_path, max_tokens=max_tokens, overlap=overlap),
            )
            if not chunks:
                skipped_count += 1
                continue

            # Embed
            texts = [c.text for c in chunks]
            embeddings = embedder.embed(texts, kind="document")

            # Store (§12.6 — per-file transaction)
            try:
                stat = fpath.stat()
                with per_file_transaction(conn):
                    # Delete old rows if reindexing
                    if reindex:
                        old = conn.execute(
                            "SELECT id FROM files WHERE path = ?", (str(fpath),),
                        ).fetchone()
                        if old:
                            old_chunks = conn.execute(
                                "SELECT id FROM chunks WHERE file_id = ?", (old[0],),
                            ).fetchall()
                            if old_chunks:
                                chunk_ids = [r[0] for r in old_chunks]
                                placeholders = ",".join("?" * len(chunk_ids))
                                conn.execute(
                                    f"DELETE FROM vec_chunks WHERE rowid IN ({placeholders})",  # noqa: S608
                                    chunk_ids,
                                )
                            conn.execute("DELETE FROM files WHERE id = ?", (old[0],))

                    file_id = insert_file(
                        conn,
                        path=str(fpath),
                        source_path=source_path,
                        file_hash=fhash,
                        mtime=stat.st_mtime,
                        size=stat.st_size,
                    )
                    total_tokens = 0
                    for chunk, emb in zip(chunks, embeddings, strict=True):
                        insert_chunk_with_embedding(conn, file_id, chunk, emb)
                        total_tokens += chunk.token_count

                elapsed_ms = int((time.monotonic() - t0) * 1000)
                indexed_count += 1
                _log_index(run_id, str(fpath), len(chunks), total_tokens, elapsed_ms)

            except Exception as exc:
                failed_count += 1
                _log_error(run_id, str(fpath), str(exc), traceback.format_exc())
                continue

        # Finalize run record
        conn.execute(
            "UPDATE index_runs SET finished_at = ?, indexed_count = ?, "
            "skipped_count = ?, failed_count = ? WHERE run_id = ?",
            (
                datetime.now(UTC).isoformat(),
                indexed_count,
                skipped_count,
                failed_count,
                run_id,
            ),
        )
        conn.commit()

        # End-of-run summary (§12.0, §12.2)
        # §12.2: failure summary prints even with --quiet
        if failed_count > 0:
            print(
                f"haku: indexed {indexed_count} files, {failed_count} failed. "
                f"See {HAKU_ROOT / 'logs' / 'errors.jsonl'}\n"
                "      Re-run `haku index` later to retry failed files.",
                file=sys.stderr,
            )
        elif not quiet:
            total_skipped = skipped_count + skipped_ext
            parts = [f"haku: indexed {indexed_count} files"]
            if total_skipped > 0:
                parts.append(f" ({total_skipped} skipped)")
            parts.append(".")
            print("".join(parts), file=sys.stderr)

        # §12.0: unsupported-extension summary prints even with --quiet if non-zero
        if skipped_ext > 0 and indexed_count == 0 and failed_count == 0:
            print(
                f"haku: indexed 0 files ({skipped_ext} skipped: unsupported extensions).\n"
                f"      supported: {_SUPPORTED_LIST}",
                file=sys.stderr,
            )

    finally:
        _release_lock(lock_fd)
        conn.close()


# ---------------------------------------------------------------------------
# Search command (§16 — vector-only in step 6)
# ---------------------------------------------------------------------------


def _normalize_scope(paths: list[str]) -> list[tuple[str, bool]]:
    """Normalize --files paths for search scoping. See §16.3.
    Returns [(resolved_path, is_dir), ...]."""
    out: list[tuple[str, bool]] = []
    for p in paths:
        resolved = str(Path(p).resolve())
        is_dir = Path(p).resolve().is_dir()
        if is_dir and not resolved.endswith("/"):
            resolved += "/"
        out.append((resolved, is_dir))
    return out


def _build_scope_clause(
    scopes: list[tuple[str, bool]],
) -> tuple[str, list[str]]:
    """Build a SQL WHERE clause for path scoping. See §16.3."""
    conditions: list[str] = []
    params: list[str] = []
    for path, is_dir in scopes:
        if is_dir:
            conditions.append("f.path LIKE ?")
            params.append(path + "%")
        else:
            conditions.append("f.path = ?")
            params.append(path)
    clause = " OR ".join(conditions)
    return f"({clause})", params


def _check_scopes_indexed(
    conn: sqlite3.Connection, scopes: list[tuple[str, bool]],
) -> None:
    """Fail if any --files scope has zero indexed files. See §16.3."""
    for path, is_dir in scopes:
        if is_dir:
            count = conn.execute(
                "SELECT COUNT(*) FROM files WHERE path LIKE ?", (path + "%",),
            ).fetchone()
        else:
            count = conn.execute(
                "SELECT COUNT(*) FROM files WHERE path = ?", (path,),
            ).fetchone()
        if count is None or count[0] == 0:
            display = path.rstrip("/") if is_dir else path
            print(
                f"haku: no indexed files under {display}\n"
                f"      run `haku index --files {display}` first.",
                file=sys.stderr,
            )
            raise SystemExit(1)


def _check_lock_warns(quiet: bool) -> None:
    """Warn if an indexer is running. See §20."""
    lock_path = HAKU_ROOT / ".lock"
    if not lock_path.exists():
        return
    fd = os.open(str(lock_path), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        if not quiet:
            print(
                "haku: indexing in progress; results reflect the corpus "
                "as of the last completed file.",
                file=sys.stderr,
            )
    finally:
        os.close(fd)


def _snippet(text: str, max_len: int = 200) -> str:
    """Truncate chunk text for display. See §17.1."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return f"...{text}..."
    return f"...{text[:max_len]}..."


def _format_text_results(
    results: list[dict[str, object]],
) -> str:
    """Format search results as human-readable text. See §17.1."""
    lines: list[str] = []
    for r in results:
        rank = r["rank"]
        score = r["score"]
        path = r["path"]
        chunk_idx = r["chunk_idx"]
        start = r["start_offset"]
        end = r["end_offset"]
        snippet = r["snippet"]
        lines.append(f"{rank}. [{score:.2f}] {path}")
        lines.append(f"   chunk {chunk_idx} · chars {start}–{end}")
        lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines)


def _format_json_results(
    query: str,
    results: list[dict[str, object]],
    took_ms: int,
    scoped_paths: list[str] | None,
    rerank: bool = False,
) -> str:
    """Format search results as JSON. See §17.2."""
    return json.dumps(
        {
            "schema_version": 1,
            "query": query,
            "took_ms": took_ms,
            "rerank": rerank,
            "scoped_paths": scoped_paths,
            "results": results,
        },
        indent=2,
    )


import sqlite_vec  # type: ignore[import-untyped]

_RRF_K = 60  # see §16.2
_RETRIEVAL_K = 50  # see §16.1 — fetch top-50 from each side


def _vec_retrieve(
    conn: sqlite3.Connection,
    q_blob: bytes,
    scope_sql: str,
    scope_params: list[str],
) -> list[tuple[int, int]]:
    """Vector ANN retrieval. Returns [(chunk_id, rank), ...]. See §16.1."""
    rows = conn.execute(
        "SELECT vc.rowid, vc.distance "
        "FROM vec_chunks vc "
        "WHERE vc.embedding MATCH ? AND vc.k = ?",
        (q_blob, _RETRIEVAL_K),
    ).fetchall()
    if not rows:
        return []

    chunk_ids = [r[0] for r in rows]
    placeholders = ",".join("?" * len(chunk_ids))

    # Apply scope filter via join
    scoped = conn.execute(
        f"SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id "  # noqa: S608
        f"WHERE c.id IN ({placeholders}){scope_sql}",
        [*chunk_ids, *scope_params],
    ).fetchall()
    scoped_ids = {r[0] for r in scoped}

    # Assign ranks by distance (ascending = best first), scoped only
    ranked = [(cid, dist) for cid, dist in rows if cid in scoped_ids]
    ranked.sort(key=lambda x: x[1])
    return [(cid, rank) for rank, (cid, _) in enumerate(ranked, 1)]


def _fts_retrieve(
    conn: sqlite3.Connection,
    query_text: str,
    scope_sql: str,
    scope_params: list[str],
) -> list[tuple[int, int]]:
    """FTS5 BM25 retrieval. Returns [(chunk_id, rank), ...]. See §16.1."""
    rows = conn.execute(
        "SELECT fc.rowid, fc.rank "
        "FROM fts_chunks fc "
        "WHERE fts_chunks MATCH ? "
        "ORDER BY fc.rank LIMIT ?",
        (query_text, _RETRIEVAL_K),
    ).fetchall()
    if not rows:
        return []

    chunk_ids = [r[0] for r in rows]
    placeholders = ",".join("?" * len(chunk_ids))

    # Apply scope filter
    scoped = conn.execute(
        f"SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id "  # noqa: S608
        f"WHERE c.id IN ({placeholders}){scope_sql}",
        [*chunk_ids, *scope_params],
    ).fetchall()
    scoped_ids = {r[0] for r in scoped}

    # Assign ranks by BM25 (already ordered), scoped only
    ranked = [(cid, bm25) for cid, bm25 in rows if cid in scoped_ids]
    return [(cid, rank) for rank, (cid, _) in enumerate(ranked, 1)]


def _rrf_fuse(
    vec_ranked: list[tuple[int, int]],
    fts_ranked: list[tuple[int, int]],
) -> dict[int, float]:
    """Reciprocal Rank Fusion. See §16.2. Returns {chunk_id: rrf_score}."""
    scores: dict[int, float] = {}
    for cid, rank in vec_ranked:
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
    for cid, rank in fts_ranked:
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
    return scores


def _fetch_chunk_details(
    conn: sqlite3.Connection, chunk_ids: list[int],
) -> dict[int, dict[str, object]]:
    """Fetch chunk + file metadata for a set of chunk IDs."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT c.id, c.chunk_idx, c.text, c.token_count, "  # noqa: S608
        f"c.start_offset, c.end_offset, f.path, f.source_path "
        f"FROM chunks c JOIN files f ON f.id = c.file_id "
        f"WHERE c.id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return {
        r[0]: {
            "chunk_id": r[0],
            "chunk_idx": r[1],
            "snippet": _snippet(r[2]),
            "token_count": r[3],
            "start_offset": r[4],
            "end_offset": r[5],
            "path": r[6],
            "source_path": r[7],
        }
        for r in rows
    }


def _cmd_search(args: argparse.Namespace) -> None:  # noqa: PLR0912, PLR0915
    quiet: bool = args.quiet
    query_text: str = args.query
    top: int = args.top
    fmt: str = args.format
    output_path: str | None = args.output

    # DB setup
    db_path = HAKU_ROOT / "database.db"
    if not db_path.exists():
        print("haku: no database found. Run `haku index` first.", file=sys.stderr)
        raise SystemExit(1)

    conn = open_db(db_path)
    try:
        check_schema_version(conn)
        _check_lock_warns(quiet)

        # Scope check (§16.3)
        scopes: list[tuple[str, bool]] | None = None
        scoped_display: list[str] | None = None
        scope_sql = ""
        scope_params: list[str] = []
        if args.files:
            scopes = _normalize_scope(args.files)
            _check_scopes_indexed(conn, scopes)
            scoped_display = [s[0].rstrip("/") for s in scopes]
            clause, scope_params = _build_scope_clause(scopes)
            scope_sql = f" AND {clause}"

        # Embed query
        t0 = time.monotonic()
        embedder = Embedder(args.embedder)
        q_vec = embedder.embed([query_text], kind="query")[0]
        q_blob = sqlite_vec.serialize_float32(q_vec)

        # Hybrid retrieval (§16.1)
        vec_ranked = _vec_retrieve(conn, q_blob, scope_sql, scope_params)
        fts_ranked = _fts_retrieve(conn, query_text, scope_sql, scope_params)

        # RRF fusion (§16.2)
        rrf_scores = _rrf_fuse(vec_ranked, fts_ranked)

        if not rrf_scores:
            took_ms = int((time.monotonic() - t0) * 1000)
            if fmt == "json":
                out = _format_json_results(query_text, [], took_ms, scoped_display)
            else:
                out = "No results found.\n"
            _emit(out, output_path)
            return

        # Sort by RRF score descending, cut to rerank_top candidates
        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        use_rerank = not args.no_rerank
        rerank_top: int = args.rerank_top
        candidate_ids = sorted_ids[:rerank_top] if use_rerank else sorted_ids[:top]

        # Fetch details for candidates
        details = _fetch_chunk_details(conn, candidate_ids)

        if use_rerank:
            # Reranking (§16.4)
            reranker = Reranker(args.rerank_model)
            passages = [
                str(details[cid]["snippet"]) for cid in candidate_ids if cid in details
            ]
            valid_ids = [cid for cid in candidate_ids if cid in details]
            rerank_scores = reranker.score(query_text, passages)

            # Re-sort by rerank score, cut to --top
            scored = list(zip(valid_ids, rerank_scores, strict=True))
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:top]

            results: list[dict[str, object]] = []
            for rank, (cid, rscore) in enumerate(scored, 1):
                entry = {**details[cid]}
                entry["score"] = round(rscore, 4)
                entry["score_kind"] = "rerank"
                entry["rank"] = rank
                results.append(entry)
            did_rerank = True
        else:
            # No reranking — use RRF scores directly
            results = []
            for rank, cid in enumerate(candidate_ids, 1):
                if cid not in details:
                    continue
                entry = {**details[cid]}
                entry["score"] = round(rrf_scores[cid], 4)
                entry["score_kind"] = "rrf"
                entry["rank"] = rank
                results.append(entry)
            did_rerank = False

        took_ms = int((time.monotonic() - t0) * 1000)

        if fmt == "json":
            out = _format_json_results(
                query_text, results, took_ms, scoped_display, did_rerank,
            )
        else:
            out = _format_text_results(results)

        _emit(out, output_path)

    finally:
        conn.close()


def _emit(text: str, output_path: str | None) -> None:
    """Write to stdout or --output file."""
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    else:
        print(text, end="" if text.endswith("\n") else "\n")


def _maybe_bootstrap(conn: sqlite3.Connection) -> None:
    """Auto-bootstrap schema if the DB is empty."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'",
    ).fetchone()
    if row is None:
        bootstrap_schema(conn)


# ---------------------------------------------------------------------------
# Init command (§6)
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> None:
    quiet: bool = args.quiet

    # Python version check (bash wrapper already enforces 3.12, but be explicit)
    import platform

    major, minor = platform.python_version_tuple()[:2]
    if int(major) < 3 or int(minor) < 12:  # noqa: PLR2004
        print(
            f"haku: Python 3.12+ required, found {platform.python_version()}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # sqlite-vec smoke test (§6.4, §7)
    db_path = HAKU_ROOT / "database.db"
    conn = open_db(db_path)
    try:
        row = conn.execute("SELECT vec_version()").fetchone()
        if not quiet:
            print(f"haku: sqlite-vec {row[0]} loaded.", file=sys.stderr)
    except Exception:
        conn.close()
        raise

    # Directories (§6.5)
    (HAKU_ROOT / "markdowns").mkdir(exist_ok=True)
    (HAKU_ROOT / "logs").mkdir(exist_ok=True)

    # Schema (§6.6)
    _maybe_bootstrap(conn)
    check_schema_version(conn)
    conn.close()

    # Next steps (§6.7)
    if not quiet:
        print(
            f"haku initialized at {HAKU_ROOT}.\n"
            f"download the ONNX models per {HAKU_ROOT}/MODELS.md, then run:\n"
            "  haku status            # verify model manifest\n"
            "  haku index --files ~/Documents",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Status command (§15)
# ---------------------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> None:
    quiet: bool = args.quiet
    db_path = HAKU_ROOT / "database.db"

    if not db_path.exists():
        print("haku: no database found. Run `haku init` first.", file=sys.stderr)
        raise SystemExit(1)

    conn = open_db(db_path)
    try:
        check_schema_version(conn)

        # Counts
        file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        db_size_bytes = db_path.stat().st_size
        db_size_mb = db_size_bytes / (1024 * 1024)
        schema_ver = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1",
        ).fetchone()[0]

        # sqlite-vec version
        vec_ver = conn.execute("SELECT vec_version()").fetchone()[0]

        # Model manifest checks (§15 — does NOT load models)
        manifest = json.loads((HAKU_ROOT / "engine" / "manifest.json").read_text())

        def _check_model(section: str, name: str) -> str:
            spec = manifest[section][name]
            model_dir = HAKU_ROOT / "models" / spec["dir"]
            for fspec in spec["files"]:
                fpath = model_dir / fspec["name"]
                if not fpath.exists():
                    return "missing"
                if fpath.stat().st_size != int(fspec["size"]):
                    return "corrupt"
                h = hashlib.sha256()
                with fpath.open("rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                if h.hexdigest() != fspec["sha256"]:
                    return "corrupt"
            return "ok"

        emb_name = manifest["defaults"]["embedder"]
        rr_name = manifest["defaults"]["reranker"]
        emb_status = _check_model("embedders", emb_name)
        rr_status = _check_model("rerankers", rr_name)

        # Last index run (§19)
        last_run = conn.execute(
            "SELECT started_at, finished_at, indexed_count, skipped_count, failed_count "
            "FROM index_runs WHERE finished_at IS NOT NULL "
            "ORDER BY run_id DESC LIMIT 1",
        ).fetchone()
        current_run = conn.execute(
            "SELECT started_at FROM index_runs WHERE finished_at IS NULL "
            "ORDER BY run_id DESC LIMIT 1",
        ).fetchone()

        # Format output (§15)
        lines = [
            f"indexed files:           {file_count:,}",
            f"db size:                 {db_size_mb:.1f} MB",
            f"schema version:          {schema_ver}",
            f"sqlite-vec version:      {vec_ver}",
            f"embedder:                {emb_status}  ({emb_name})",
            f"reranker:                {rr_status}  ({rr_name})",
        ]
        if last_run:
            lines.append(f"last index started:      {last_run[0]}")
            lines.append(f"last index finished:     {last_run[1]}")
            lines.append(
                f"last run: indexed/skipped/failed:  "
                f"{last_run[2]} / {last_run[3]} / {last_run[4]}",
            )
        else:
            lines.append("last index started:      -")
            lines.append("last index finished:     -")
        lines.append(
            f"current run (if active): {current_run[0] if current_run else '-'}",
        )

        out = "\n".join(lines)
        if not quiet:
            _emit(out, args.output)

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Purge command (§18)
# ---------------------------------------------------------------------------


def _cmd_purge(args: argparse.Namespace) -> None:
    quiet: bool = args.quiet
    db_path = HAKU_ROOT / "database.db"

    if not db_path.exists():
        print("haku: no database found. Run `haku init` first.", file=sys.stderr)
        raise SystemExit(1)

    conn = open_db(db_path)
    lock_fd = _acquire_lock()

    try:
        check_schema_version(conn)

        # Step 1–2: find dead files (§18)
        rows = conn.execute("SELECT id, path FROM files").fetchall()
        dead_file_ids: list[int] = []
        for fid, fpath in rows:
            if not Path(fpath).exists():
                dead_file_ids.append(fid)

        if not dead_file_ids:
            if not quiet:
                print("haku: nothing to purge.", file=sys.stderr)
            return

        # Step 3: collect dead chunk ids BEFORE any delete (§18)
        placeholders = ",".join("?" * len(dead_file_ids))
        dead_chunk_rows = conn.execute(
            f"SELECT id FROM chunks WHERE file_id IN ({placeholders})",  # noqa: S608
            dead_file_ids,
        ).fetchall()
        dead_chunk_ids = [r[0] for r in dead_chunk_rows]

        # Step 4: delete from vec_chunks (§18)
        if dead_chunk_ids:
            vc_placeholders = ",".join("?" * len(dead_chunk_ids))
            conn.execute(
                f"DELETE FROM vec_chunks WHERE rowid IN ({vc_placeholders})",  # noqa: S608
                dead_chunk_ids,
            )

        # Step 5: delete from files (CASCADE deletes chunks, trigger cleans FTS)
        conn.execute(
            f"DELETE FROM files WHERE id IN ({placeholders})",  # noqa: S608
            dead_file_ids,
        )

        # Step 6: commit
        conn.commit()

        if not quiet:
            print(
                f"haku: purged {len(dead_file_ids)} file(s), "
                f"{len(dead_chunk_ids)} chunk(s).",
                file=sys.stderr,
            )

    finally:
        _release_lock(lock_fd)
        conn.close()


# ---------------------------------------------------------------------------
# Version command (§11)
# ---------------------------------------------------------------------------


def _cmd_version(args: argparse.Namespace) -> None:
    full: bool = args.full

    manifest = json.loads((HAKU_ROOT / "engine" / "manifest.json").read_text())
    emb_name = manifest["defaults"]["embedder"]
    rr_name = manifest["defaults"]["reranker"]
    emb = manifest["embedders"][emb_name]
    rr = manifest["rerankers"][rr_name]

    import sqlite3 as _sqlite3

    import onnxruntime as _ort  # type: ignore[import-untyped]

    sqlite_ver = _sqlite3.sqlite_version
    db_path = HAKU_ROOT / "database.db"
    schema_ver = "-"
    vec_ver = "-"
    if db_path.exists():
        conn = open_db(db_path)
        try:
            row = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1",
            ).fetchone()
            if row:
                schema_ver = str(row[0])
            vec_row = conn.execute("SELECT vec_version()").fetchone()
            if vec_row:
                vec_ver = str(vec_row[0])
        finally:
            conn.close()

    def _trunc(val: str, length: int = 8) -> str:
        return val[:length] if not full else val

    def _file_sha(section: str, name: str, fname: str) -> str:
        for f in manifest[section][name]["files"]:
            if f["name"] == fname:
                return _trunc(f["sha256"])
        return "?"

    if full:
        lines = [
            "haku 0.1.0",
            f"schema:        {schema_ver}",
            f"sqlite:        {sqlite_ver}",
            f"sqlite-vec:    {vec_ver}",
            f"onnxruntime:   {_ort.__version__}",
            f"embedder:      {emb_name}",
            f"  upstream:    {emb['upstream']}",
            f"  revision:    {_trunc(emb['revision'])}",
            f"  model.onnx sha:     {_file_sha('embedders', emb_name, 'model.onnx')}",
            f"  tokenizer.json sha: {_file_sha('embedders', emb_name, 'tokenizer.json')}",
            f"reranker:      {rr_name}",
            f"  upstream:    {rr['upstream']}",
            f"  revision:    {_trunc(rr['revision'])}",
            f"  model.onnx sha:     {_file_sha('rerankers', rr_name, 'model.onnx')}",
            f"  tokenizer.json sha: {_file_sha('rerankers', rr_name, 'tokenizer.json')}",
            f"licenses:      see {HAKU_ROOT}/LICENSES.md  (note: PyMuPDF4LLM is AGPLv3)",
        ]
    else:
        lines = [
            "haku 0.1.0",
            f"schema:        {schema_ver}",
            f"sqlite:        {sqlite_ver}",
            f"sqlite-vec:    {vec_ver}",
            f"onnxruntime:   {_ort.__version__}",
            f"embedder:      {emb_name} @ {emb['upstream']}",
            f"embedder rev:  {_trunc(emb['revision'])}",
            f"embedder sha:  {_file_sha('embedders', emb_name, 'model.onnx')}"
            f"  ({emb_name}/model.onnx)",
            f"reranker:      {rr_name} @ {rr['upstream']}",
            f"reranker rev:  {_trunc(rr['revision'])}",
            f"reranker sha:  {_file_sha('rerankers', rr_name, 'model.onnx')}"
            f"  ({rr_name}/model.onnx)",
            f"licenses:      see {HAKU_ROOT}/LICENSES.md  (note: PyMuPDF4LLM is AGPLv3)",
        ]

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Argument parsing (§9)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haku", description="Local semantic search CLI.")
    parser.add_argument("--version", action="store_true", help="Print version info.")
    parser.add_argument("--full", action="store_true", help="Full hashes with --version.")
    sub = parser.add_subparsers(dest="command")

    idx = sub.add_parser("index", help="Index files for search.")
    idx.add_argument("--files", nargs="+", required=True, help="Files or directories to index.")
    idx.add_argument("--chunks", type=int, default=512, help="Max tokens per chunk.")
    idx.add_argument("--overlap", type=int, default=64, help="Overlap tokens across chunks.")
    idx.add_argument("--embedder", default=None, help="Embedder model directory name.")
    idx.add_argument("--reindex", action="store_true", help="Force re-embed all files.")
    idx.add_argument("--format", choices=["json", "text"], default="text")
    idx.add_argument("--output", default=None, help="Write output to file.")
    idx.add_argument("--quiet", action="store_true", help="Suppress progress and non-error stderr.")

    srch = sub.add_parser("search", help="Search indexed files.")
    srch.add_argument("query", help="Search query text.")
    srch.add_argument("--files", nargs="+", default=None, help="Scope search to paths.")
    srch.add_argument("--top", type=int, default=5, help="Number of results to return.")
    srch.add_argument("--rerank-top", type=int, default=20, help="Candidates for reranker.")
    srch.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking.")
    srch.add_argument("--rerank-model", default=None, help="Reranker model directory name.")
    srch.add_argument("--embedder", default=None, help="Embedder model directory name.")
    srch.add_argument("--format", choices=["json", "text"], default="text")
    srch.add_argument("--output", default=None, help="Write output to file.")
    srch.add_argument("--quiet", action="store_true", help="Suppress non-error stderr.")

    init = sub.add_parser("init", help="Initialize haku (venv, DB, dirs).")
    init.add_argument("--quiet", action="store_true", help="Suppress non-error output.")

    status = sub.add_parser("status", help="Show index and model status.")
    status.add_argument("--format", choices=["json", "text"], default="text")
    status.add_argument("--output", default=None, help="Write output to file.")
    status.add_argument("--quiet", action="store_true", help="Suppress output.")

    purge = sub.add_parser("purge", help="Remove DB entries for deleted files.")
    purge.add_argument("--format", choices=["json", "text"], default="text")
    purge.add_argument("--output", default=None, help="Write output to file.")
    purge.add_argument("--quiet", action="store_true", help="Suppress output.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        _cmd_version(args)
    elif args.command == "index":
        _cmd_index(args)
    elif args.command == "search":
        _cmd_search(args)
    elif args.command == "init":
        _cmd_init(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "purge":
        _cmd_purge(args)
    elif args.command is None:
        parser.print_help()
        raise SystemExit(1)
    else:
        print(f"haku: unknown command '{args.command}'", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
