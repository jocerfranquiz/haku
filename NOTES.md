
## Step 1 — tokenizer + tooling
- **Date:** 2026-05-26
- **Shipped:** `engine/tokenizer.py` (encode/decode/count), 4 roundtrip tests, ruff/mypy/pytest tooling, pre-commit config. `make check` passes.
- **Surprised me:** Qwen3 tokenizer emits a special token (id 151643) on empty-string input — adjusted the empty-string test to only assert `decode([]) == ""` instead of assuming `encode("") == []`.
- **Deferred:** nothing
- **Open questions for Luis:** The `tokenizers` package (v0.23.1) ships its own type stubs, so no `type: ignore` was needed — worth noting if you expected those annotations from the DESIGN.md sketch.

## Step 2 — chunk.py + EN/ES golden tests
- **Date:** 2026-05-26
- **Shipped:** `engine/chunk.py` (~95-line splitter), 11 golden-file tests (6 EN, 4 ES, 1 edge-case pair), fixture docs in `engine/fixtures/`. `make check` passes (15 total tests).
- **Surprised me:** DESIGN.md §13 uses `from typing import Iterable` but ruff UP035 requires `from collections.abc import Iterable` on Python 3.12 — minor, same behavior.
- **Deferred:** nothing
- **Open questions for Luis:** The `engine/fixtures/` directory isn't in DESIGN.md §4's layout. It's test-only data. Let me know if you'd prefer fixtures inline in the test file instead.

## Step 3 — storage.py schema + version check + tx helper
- **Date:** 2026-05-26
- **Shipped:** `engine/storage.py` (full §19 schema, sqlite-vec loading, version-mismatch refusal, `per_file_transaction`, `insert_chunk_with_embedding`), 11 tests covering schema bootstrap, WAL, vec extension, version check, atomic insert, rollback, and FTS trigger sync. `make check` passes (26 total tests).
- **Surprised me:** ruff UP017 wants `datetime.UTC` but mypy 2.1.0 doesn't recognize it as a class attribute — using `from datetime import UTC` resolves both.
- **Deferred:** nothing
- **Open questions for Luis:** The `sqlite_vec` package (v0.1.9) has no type stubs, so `import sqlite_vec` needs a `type: ignore[import-untyped]`. This is the kind of boundary CLAUDE.md anticipated for SQLite/ONNX deps.

## Step 4 — embed.py ONNX embedder + manifest check
- **Date:** 2026-05-26
- **Shipped:** `engine/embed.py` (lazy ONNX session, manifest SHA-256 check at load, last-token pooling, L2-normalize, query/document prefix asymmetry, batched embedding), 11 tests (5 manifest checks + 6 embedding). `make check` passes (37 total tests).
- **Surprised me:** The `onnx-community` Qwen3 export includes KV cache inputs (`past_key_values.N.key/value` for 28 layers) and `position_ids` — it's a generative causal LM export, not a pure encoder. Had to supply empty KV cache tensors (shape `batch, 8, 0, 128`) and position_ids for embedding use. Only requesting `last_hidden_state` output avoids materializing the KV cache outputs.
- **Deferred:** nothing
- **Open questions for Luis:** The model architecture constants (28 layers, 8 KV heads, 128 head_dim) are hardcoded in `embed.py`. These could be derived from the model's input metadata at session creation, but hardcoding is simpler and these are fixed for this model. Worth revisiting if model swapping becomes real.

## Step 5 — end-to-end haku index
- **Date:** 2026-05-26
- **Shipped:** `engine/haku.py` (dispatcher + index subcommand), `haku` bash entrypoint, `engine/generate_corpus.py` (test corpus generator). Full indexing pipeline: file discovery (§12.0), extension handlers (.txt/.md/.pdf/.html/.docx), incremental skip check (§12.1), ThreadPoolExecutor for PDF conversion (§12.5), per-file transactions (§12.6), JSONL logging (§22), tqdm progress (§12.7), end-of-run summary. 9 end-to-end tests. `make check` passes (46 total tests, ~77s).
- **Surprised me:** `Path(__file__).resolve()` follows symlinks, which broke the test fixture (temp dir with symlinked engine/). Fixed by having `haku.py` read `HAKU_HOME` from env (the bash wrapper exports it) and fall back to `__file__`-based resolution. This is a minor deviation from §14's "HAKU_HOME is read only by the bash wrapper" — in practice the wrapper exports it and haku.py uses it for DB/logs/markdowns paths.
- **Deferred:** `--format json` output for index (§17) — not exercised yet; will matter in step 9 when the full CLI surface is built.
- **Open questions for Luis:** The test suite now takes ~77s because each end-to-end test spins up a subprocess that loads the ONNX model. If this becomes painful, we could share the model across tests or mark the slow tests. Also: `generate_corpus.py` is excluded from mypy because `pymupdf` and `python-docx` lack type stubs — it's a dev-only script, not runtime code.

## Step 6 — haku search (vector-only)
- **Date:** 2026-05-26
- **Shipped:** `haku search` subcommand with vector-only retrieval via sqlite-vec ANN, `--top` limiting, `--files` path scoping (§16.3), text and JSON output formats (§17), `--output` file write, lock-check warning (§20), un-indexed scope error. 9 search tests. `make check` passes (55 total tests, ~122s).
- **Surprised me:** Search tests use a module-scoped fixture that indexes once and shares the DB across all 9 tests — cuts test time vs. per-test indexing. The `scope="module"` on the fixture is key.
- **Deferred:** FTS5 lexical search + RRF fusion (step 7), reranking (step 8). Score is currently cosine similarity from vector-only; step 7 replaces with RRF score.
- **Open questions for Luis:** The cosine similarity conversion from sqlite-vec L2 distance (`1 - dist²/2`) assumes L2-normalized vectors, which we guarantee in embed.py. If you ever see scores > 1.0 or negative, it means a vector wasn't normalized — worth sanity-checking on your test runs.

## Step 7 — FTS5 + RRF hybrid search
- **Date:** 2026-05-26
- **Shipped:** Hybrid search: vector ANN (sqlite-vec) + FTS5 BM25 retrieval, fused with RRF (k=60, §16.2). Refactored `_cmd_search` into `_vec_retrieve`, `_fts_retrieve`, `_rrf_fuse`, `_fetch_chunk_details` helpers. `score_kind` is now `"rrf"`. 7 new hybrid tests (diacritics folding, exact keyword FTS, scoping on both paths). All 62 tests pass (~2.7min).
- **Surprised me:** Nothing — FTS5 triggers from step 3 worked out of the box, `unicode61 remove_diacritics 2` correctly folds `café`→`cafe` as designed.
- **Deferred:** Reranking (step 8). Currently the RRF top-N goes straight to output; step 8 inserts the cross-encoder between RRF and final output.
- **Open questions for Luis:** none

## Step 8 — bge-reranker-v2-m3 cross-encoder
- **Date:** 2026-05-27
- **Shipped:** `Reranker` class in `embed.py` (lazy ONNX session, manifest SHA check, batch scoring of (query, passage) pairs), wired into `_cmd_search` after RRF fusion. Reranking on by default; `--no-rerank` opts out. `score_kind` is `"rerank"` when active, `"rrf"` when skipped. JSON output `"rerank"` field reflects actual state. 5 new rerank tests (default on, opt-out, A/B ordering, Spanish). Updated 2 older tests for compatibility. All 67 tests pass (~4.7min).
- **Surprised me:** The bge-reranker-v2-m3 ONNX export is a clean cross-encoder — just `input_ids` + `attention_mask` → single logit per pair. No KV cache, no position_ids. Much simpler than the Qwen3 embedder export. Also: had to fix a typo in manifest.json (`"17082900S"` → `"17082900"`).
- **Deferred:** nothing
- **Open questions for Luis:** The reranker uses its own tokenizer.json (loaded from `models/bge-reranker-v2-m3/tokenizer.json`), separate from the shared Qwen tokenizer in `tokenizer.py`. This is correct — different models have different vocabularies.

## Step 9 — init, status, purge, --version, --quiet
- **Date:** 2026-05-27
- **Shipped:** Four new commands: `haku init` (venv bootstrap via bash wrapper + DB/dirs via Python, §6), `haku status` (model manifest check + DB stats, §15), `haku purge` (ordered deletion per §18, takes lock per §20), `haku --version` / `--version --full` (§11). `--quiet` wired across all commands. Bash wrapper handles the chicken-and-egg: creates `.venv` + installs deps before Python runs. 11 new CLI tests. All 78 tests pass (~5.5min).
- **Surprised me:** Python check for ≥3.12 inside Python is trivially true since we're already running 3.12. The real gate is the bash wrapper's `command -v python3.12` check. Also: `--version` is a top-level flag (not a subcommand) per §11 — `haku --version`, not `haku version`.
- **Deferred:** nothing
- **Open questions for Luis:** none — full CLI surface from §9 is now implemented.
