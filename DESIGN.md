# `haku` — Multilingual Semantic Searcher

> *Haku* = "search" in Finnish. A 100% local, 100% open-source, 100% lightweight semantic search CLI for text, markdown, and PDFs.

## Contents

- **Part I — Orientation** (§1–§8): goals, licensing, architecture, file layout, memory budget, first-run bootstrap, sqlite-vec install, bash entrypoint.
- **Part II — Using `haku`** (§9–§11): CLI surface, config + precedence, `--version`.
- **Part III — How it works** (§12–§18): indexing pipeline, chunker, tokenizer, models & manifest, search pipeline, output formats, `purge`.
- **Part IV — Storage & operations** (§19–§23): SQLite schema, concurrency contract, markdown cache, logging, backup & restore.
- **Part V — Planning** (§24–§25): risks, build order.

---

# Part I — Orientation

## 1. Goals & Non-Goals

### Goals
- Local semantic + lexical hybrid search over personal documents.
- **Supported extensions** (v1): `.txt .md .markdown .mdx .pdf .html .docx`. See §12.0.
- Bilingual quality: **English + Spanish** out of the box.
- Single-binary feel: one `haku` bash entrypoint, transparent venv activation.
- CPU-only execution (no GPU dependency).
- Incremental indexing: never re-embed unchanged files.

### Non-Goals
- No frontier-model API calls.
- No GPU support in v1 (CPU-only by design).
- No daemon / no web UI / no network listener.
- No multi-user concurrency beyond a single-machine indexer lock.

---

## 2. Licensing

- **`haku` itself**: GPL v3.
- **Runtime dependency note**: `PyMuPDF4LLM` is **AGPL v3**. GPLv3 and AGPLv3 are compatible for combination, but the AGPL network clause propagates to the combined work. Since `haku` is a local CLI (no network service), this is fine in practice.
- **Model weights have their own licenses**, tracked in `LICENSES.md`:
  - `Qwen3-Embedding-0.6B` — Apache 2.0
  - `bge-reranker-v2-m3` — MIT
  Both are permissive and combine cleanly with GPLv3 distribution. The
  ONNX-converted weights we ship are downloaded from the
  `onnx-community/*` Hugging Face repos (see `MODELS.md`) and inherit
  the upstream licenses as derivative works. Users who swap in a
  different fine-tune or a different ONNX export are responsible for
  verifying license compatibility.
- A **`LICENSES.md`** file ships with the project listing every runtime dep and model weight with its license. The `haku --version` command prints a one-line summary plus a pointer to that file.

---

## 3. Architecture Overview

```
┌──────────────────┐
│  haku (bash)     │  exec /haku/.venv/bin/python /haku/engine/haku.py "$@"
└────────┬─────────┘
         │
┌────────▼──────────────────────────────────────────────┐
│  haku.py  (single dispatcher)                         │
│  ├─ subcommands: index | search | status | purge      │
│  ├─ shared: config, logging, db handle, tokenizer     │
│  └─ uses modules below                                │
└───┬──────────┬──────────┬──────────┬──────────────────┘
    │          │          │          │
┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼─────────┐
│chunk.py│ │embed.py│ │storage │ │tokenizer.py │
│        │ │ (ONNX) │ │  .py   │ │ (shared)    │
└────────┘ └────────┘ └────────┘ └─────────────┘
```
---

### Why one dispatcher, not five scripts
- Shared imports (tokenizer, config, DB) load once.
- Easier to navigate and refactor.
- Bash wrapper stays trivial: `exec ... haku.py "$@"`.

---

## 4. File Layout

```
/haku/
├── haku                     # bash entrypoint
├── config.json              # default config
├── database.db              # sqlite + sqlite-vec + FTS5
├── .venv/                   # python virtualenv
├── .lock                    # fcntl indexer lock
├── engine/
│   ├── haku.py              # single dispatcher
│   ├── chunk.py             # ~95-line markdown splitter
│   ├── embed.py             # ONNX Runtime embedder + reranker
│   ├── storage.py           # schema, migrations, CRUD
│   ├── tokenizer.py         # shared Qwen tokenizer wrapper
│   └── manifest.json        # model file hashes + sizes (shipped)
├── models/                  # user-placed .onnx files
│   ├── qwen3-embedding-0.6b/
│   └── bge-reranker-v2-m3/
├── markdowns/               # cached PDF→MD conversions
└── logs/
    ├── index.jsonl
    └── errors.jsonl
```

---

## 5. Memory Budget

Rough peak RSS during the heaviest operation (indexing with both models cached) on a 16 GB laptop. **These are approximations**: the per-component figures are nominal steady-state values, the "Total peak" rounds up modestly for OS overhead and fragmentation, and an unusually large input document (a 500-page PDF, a `.docx` with many embedded images) can push PyMuPDF4LLM/mammoth working buffers past the listed value. The numbers also assume the specific `onnx-community` INT8 builds we pin in `MODELS.md`; a different community quantization (UINT8, mixed-precision, FP16 fallback) will shift the in-memory model line items by 100–300 MB. The numbers are a planning aid, not a contract — confirm with `/usr/bin/time -v` on representative inputs before quoting them anywhere user-facing.

| Component | Peak RAM |
|-----------|----------|
| Python interpreter + stdlib | ~30 MB |
| ONNX Runtime (CPU EP) + arenas | ~150 MB |
| Qwen3-Embedding-0.6B INT8 (in-memory) | ~350 MB |
| bge-reranker-v2-m3 INT8 (in-memory) | ~330 MB |
| Tokenizer (Rust `tokenizers`) | ~50 MB |
| sqlite-vec arenas + SQLite cache | ~100 MB |
| PyMuPDF4LLM working buffers (large PDFs) | ~200 MB |
| Embedding batch + scratch | ~100 MB |
| **Total peak** | **~1.3–1.5 GB** |

**Search-only mode** (no indexing): drop PyMuPDF4LLM and shrink to ~900 MB peak; with `--no-rerank`, ~600 MB.

`haku` is intended to run comfortably on machines with 8 GB RAM. The reranker is the single biggest line item — users on tight memory budgets should run with `--no-rerank` or skip it via `config.json` (`"rerank": false`).

---

## 6. First-Run Bootstrap: `haku init`

The bash wrapper (§8) assumes `/haku/.venv` and `/haku/database.db` already exist. `haku init` creates them.

### What it does

1. **Sanity-check Python**: confirm `python3 --version` ≥ 3.10. Refuse with a clear message otherwise.
2. **Create `/haku/.venv`**: `python3 -m venv /haku/.venv`.
3. **Install deps into the venv**: `pip install -r /haku/engine/requirements.txt` — includes `onnxruntime`, `sqlite-vec`, `tokenizers`, `pymupdf4llm`, `selectolax`, `mammoth`, `tqdm`.
4. **Smoke-test sqlite-vec extension loading**: open a throwaway DB and call `SELECT vec_version()`. If extension loading fails (§7), abort with the documented remediation message.
5. **Create directories**: `/haku/markdowns/`, `/haku/logs/` if absent.
6. **Initialize the database**: open `/haku/database.db`, set `PRAGMA journal_mode=WAL`, run all `CREATE TABLE` / `CREATE TRIGGER` statements from §19, insert `schema_version (version=1, applied_at=NOW)`.
7. **Print next steps**:
   ```
   haku initialized at /haku.
   download the ONNX models per /haku/MODELS.md, then run:
     haku status            # verify model manifest
     haku index --files ~/Documents
   ```

### What it does NOT do

- Download models. Users **manually place** ONNX files in `/haku/models/<name>/` per §15. This is intentional: model weights are large (~600 MB total), license-sensitive, and shouldn't be silently fetched.
- Edit shell rc files. Adding `/haku/haku` to `PATH` is the user's responsibility.

### Idempotency

`haku init` is safe to re-run. If `.venv` exists, deps are re-resolved (`pip install` skips already-satisfied packages — **but** this only holds when `requirements.txt` uses exact pins (`==`); loose constraints like `>=` allow newer transitive deps to be pulled in on re-runs, so `requirements.txt` ships with strict `==` pins on every line for true idempotency). If `database.db` exists, the schema is validated against `EXPECTED_SCHEMA_VERSION`; mismatch refuses to run with the §19 message. The directory `mkdir`s are `exist_ok=True`. **Never destructive** — no flag deletes user data; `haku init --force` is deliberately not provided.

Implementation lives in `engine/haku.py` like any other subcommand; the only externally-visible difference is that it's the one command that must run *before* the others.

## 7. sqlite-vec Installation

`sqlite-vec` is a loadable SQLite extension. **No system install required** — the pip package bundles the compiled binary for the host platform:

```
pip install sqlite-vec
```

Loaded from Python:

```python
import sqlite3, sqlite_vec
conn = sqlite3.connect("/haku/database.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)
```

**Caveat**: Python's stock `sqlite3` module needs to be built with extension loading enabled. Ubuntu, Debian, Fedora, and macOS ship it enabled by default. Some minimal Docker base images (`python:3.X-alpine`, certain `slim` variants) disable it.

`haku status` runs a smoke test (`SELECT vec_version()`) and reports the version. If extension loading fails, status prints a clear remediation message:

```
haku: sqlite-vec failed to load: the system Python's sqlite3 module was built
      without --enable-loadable-sqlite-extensions. Reinstall Python (pyenv,
      uv, or a distribution that ships it enabled) and recreate /haku/.venv.
```

Exit non-zero. No graceful degradation — vector search is core functionality.

---

## 8. Bash Entrypoint

```bash
#!/usr/bin/env bash
# /haku/haku
set -euo pipefail
HAKU_HOME="${HAKU_HOME:-/haku}"
exec "$HAKU_HOME/.venv/bin/python" "$HAKU_HOME/engine/haku.py" "$@"
```

`exec` replaces the shell process — no subshell state leaks, no `source activate` side effects. The venv's Python already knows its own `sys.path`; activation is unnecessary when invoking the interpreter directly.

---


# Part II — Using `haku`

## 9. CLI Surface

`--format`, `--output`, and `--quiet` are **global** flags (accepted by every subcommand). Subcommand-specific flags follow.

```
haku index   [--files PATH ...] [--chunks 512] [--overlap 64]
             [--embedder NAME] [--reindex]
             [--format json|text] [--output PATH] [--quiet]
haku search  QUERY [--files PATH ...] [--top 5] [--rerank-top 20]
             [--no-rerank] [--rerank-model NAME] [--embedder NAME]
             [--format json|text] [--output PATH] [--quiet]
haku status  [--format json|text] [--output PATH] [--quiet]
haku purge   [--format json|text] [--output PATH] [--quiet]
haku init    [--quiet]
haku --version [--full]
haku --help
```

Notes:
- `--embedder` and `--rerank-model` accept a **directory name** inside `/haku/models/` (not a path). The directory must be declared in `engine/manifest.json` (§15); unknown names fail with `unknown model 'X'; declare it in engine/manifest.json`.
- `--format` applies to every command's stdout/`--output`. For `index` and `purge`, `json` emits a run-summary object; `text` emits human-readable lines.

### Flag semantics
| Flag             | Default     | Scope   | Notes |
|------------------|-------------|---------|-------|
| `--chunks`       | 512 tokens  | index   | Token cap per chunk |
| `--overlap`      | 64 tokens   | index   | Overlap across boundaries |
| `--top`          | 5           | search  | Final result count after rerank |
| `--rerank-top`   | 20          | search  | Candidates passed to reranker post-RRF |
| `--no-rerank`    | off         | search  | Skip cross-encoder, return RRF order |
| `--reindex`      | off         | index   | Force rebuild (ignores hash check) |
| `--files`        | all indexed | index/search | `rm`-style: one or more paths, each a file or directory (recursed). At index time = source set. At search time = re-scoping filter via SQL prefix match (§16.3). |
| `--embedder`     | from config | index/search | Directory name inside `/haku/models/`; must be in `engine/manifest.json` |
| `--rerank-model` | from config | search  | Directory name inside `/haku/models/`; must be in `engine/manifest.json` |
| `--format`       | `text`      | global  | `json` for scripting / piping (§17) |
| `--output`       | stdout      | global  | Write to file instead of stdout |
| `--quiet`        | off         | global  | Suppresses tqdm + non-error stderr |

`--files` examples:
```
haku index  --files ~/books/cs ~/books/math ~/papers/foo.pdf
haku search "turing test" --files ~/books/cs        # only files under ~/books/cs/
```

### Config precedence (highest wins)
**CLI flag > environment variable (`HAKU_*`) > `config.json` > built-in defaults.**

---

## 10. Config (`/haku/config.json` defaults)

```json
{
  "chunks": 512,
  "overlap": 64,
  "top": 5,
  "rerank_top": 20,
  "rerank": true,
  "embedder_dir": "models/qwen3-embedding-0.6b",
  "reranker_dir": "models/bge-reranker-v2-m3",
  "embedding_batch_size": 16,
  "onnx_intra_op_threads": 0,
  "format": "text",
  "quiet": false
}
```

Env-var override pattern: `HAKU_CHUNKS=256 haku index ...` overrides `config.json`; a `--chunks 128` CLI flag overrides the env var.

**Notes on individual keys**:
- `onnx_intra_op_threads: 0` — passed to `SessionOptions.intra_op_num_threads`. `0` means "let ONNX Runtime choose" (typically `min(physical_cores, 4)`). Set to a positive integer to pin.
- `embedding_batch_size`: bigger batches are faster on CPU up to a point, then memory-bound. 16 is conservative; 32 is reasonable on a 16 GB laptop.

---

## 11. `haku --version`

```
haku 0.1.0
schema:        1
sqlite:        3.45.1
sqlite-vec:    0.1.9
onnxruntime:   1.18.0
embedder:      qwen3-embedding-0.6b @ onnx-community/Qwen3-Embedding-0.6B-ONNX
embedder rev:  6a3f2c1d
embedder sha:  abc12345  (qwen3-embedding-0.6b/model.onnx)
reranker:      bge-reranker-v2-m3 @ onnx-community/bge-reranker-v2-m3-ONNX
reranker rev:  6f5ff652
reranker sha:  ghi67890  (bge-reranker-v2-m3/model.onnx)
licenses:      see /haku/LICENSES.md  (note: PyMuPDF4LLM is AGPLv3)
```

Useful for bug reports and reproducibility. The `upstream` repo and
`revision` come from `engine/manifest.json` (§15). The SHA-256 hashes
and revision IDs are **truncated to 8 hex chars** by default for human
scannability. Bug reports often need the full values to disambiguate a
corrupted file from a legitimate model swap — `haku --version --full`
prints the complete 64-char SHA-256s and 40-char revision SHAs:

```
embedder:      qwen3-embedding-0.6b
  upstream:    onnx-community/Qwen3-Embedding-0.6B-ONNX
  revision:    6a3f2c1d... (40 hex)
  model.onnx sha:     abc12345def67890...0123ffee
  tokenizer.json sha: fedcba98...4321abcd
reranker:      bge-reranker-v2-m3
  upstream:    onnx-community/bge-reranker-v2-m3-ONNX
  revision:    6f5ff652... (40 hex)
  model.onnx sha:     ghi67890...beef1234
  tokenizer.json sha: 1122aabb...3344ccdd
```

`--full` is the only flag accepted by `--version`; everything else is rejected.

---


# Part III — How it works

## 12. Indexing Pipeline

```
file → (PDF? → PyMuPDF4LLM → cached .md) → chunker → tokenizer → embedder → SQLite
                                                                              ├─ files
                                                                              ├─ chunks
                                                                              ├─ vec_chunks
                                                                              └─ fts_chunks
```

### Per-file sequence

Each file is processed independently: discovery → skip check → extract → chunk → embed → single per-file transaction. A failure at any point before `COMMIT` rolls back the file's data entirely, so the corpus only ever advances by complete files.

---

### 12.0 File discovery & extension handling

`--files` expands each argument:
- A file path → indexed if extension is supported.
- A directory path → walked recursively; every file with a supported extension is indexed.

**Supported extensions and handlers**:

| Ext | Handler | Dep |
|-----|---------|-----|
| `.txt` | read as UTF-8, treat as plain markdown | — |
| `.md`, `.markdown`, `.mdx` | read as UTF-8 (MDX JSX tags pass through as noise) | — |
| `.pdf` | `PyMuPDF4LLM` → cached `.md` | `pymupdf4llm` (AGPL) |
| `.html` | strip tags → markdown-ish text | `selectolax` |
| `.docx` | extract text + headings → markdown | `mammoth` |

Unsupported extensions (anything not in the table, including `.rst`, `.org`, images, archives, source code) are **silently skipped at discovery time** — no log entry, no error. This keeps log noise low when a user points `--files` at a mixed directory like `~/Documents`.

However, the **end-of-run summary always reports the count** (even with `--quiet` if non-zero), so a user pointing `--files` at a directory of `.rst` files and getting zero indexed doesn't see silent failure:

```
haku: indexed 0 files (124 skipped: unsupported extensions).
      supported: .txt .md .markdown .mdx .pdf .html .docx
```

Extension handler failures (corrupt PDF, malformed DOCX, unparseable HTML) follow a **uniform skip-and-log policy** detailed in §12.2: no DB row, entry in `errors.jsonl`, end-of-run stderr summary, auto-retry on next `haku index`.

### File lifecycle

Most files cycle between `Indexed` and `Skipped` across runs: once committed, the hash check (§12.1) short-circuits subsequent runs. Edits, deletions, and handler failures each have their own well-defined exit edge; nothing is ever orphaned in DB without a recovery path (`haku purge` reconciles disk vs. DB).

---

### 12.1 Incremental skip check
For each file: compute `sha256(path || mtime || size)`. If the DB has a row in `files` with the same hash, skip the file entirely (no PDF conversion, no chunking, no embedding). `--reindex` bypasses this.

### 12.2 PDF handling
- Convert with `PyMuPDF4LLM` to `/haku/markdowns/<sha256-of-path>.md`.
- On exception:
  - Log structured entry to `logs/errors.jsonl` (path, error, traceback).
  - Increment in-run failed counter.
  - Do **not** insert a row into `files` — the failed file stays "unindexed" so a later `haku index` retry picks it up naturally (no `--reindex` needed).
  - Continue to the next file.
- At the end of an index run, if `failed > 0`, print a clear summary to stderr (even with `--quiet`):
  ```
  haku: indexed 4210 files, 7 failed. See /haku/logs/errors.jsonl
        Re-run `haku index` later to retry failed files.
  ```
- Re-use cached markdown on subsequent indexes (key = source path hash). Cached markdown is only written after successful conversion, so partial files never pollute the cache.

### 12.3 Chunking (custom ~95-line splitter)
See section 11 for the full implementation. Algorithm:
1. Split on `^## ` (H2 boundaries) first.
2. Within each section, split on `\n\n` (paragraph boundaries).
3. Token-pack paragraphs up to `--chunks` (512) using the Qwen tokenizer.
4. Apply `--overlap` (64 tokens) **only across chunk boundaries**, never inside.

No LlamaIndex dep. The custom splitter is small enough to audit in one screen.

### 12.4 Embedding
- Runtime: **ONNX Runtime (CPU EP)**.
- Model: **Qwen3-Embedding-0.6B**, exported to ONNX with **INT8 dynamic quantization** via `optimum-cli`.
  - Footprint: ~300 MB on disk (vs. ~600 MB FP16, ~1.2 GB FP32).
  - Speed: 2–3× faster than FP16 on CPU.
  - Quality: published benchmarks show <1% recall@10 drop vs. FP16 on retrieval tasks.
- Batch size: configurable (default 16 on CPU).
- Last-token pool + L2-normalize per the upstream Qwen3-Embedding
  recipe. (Qwen3 is a causal LM; the sequence embedding lives in the
  final token's hidden state, not the mean of all tokens.)
- Queries are wrapped with the instruction prefix
  `"Instruct: <task>\nQuery: <text>"` before embedding; documents are
  embedded as-is. `embed.py` exposes a `kind: "query" | "document"`
  argument to enforce this asymmetry. The task string defaults to
  `"Given a web search query, retrieve relevant passages that answer
  the query"` per the upstream recipe.
- Output dim: **1024** (Qwen3-Embedding-0.6B native; unchanged by quantization).

### 12.5 Concurrency & locking
- **One indexer at a time**: `fcntl.flock` on `/haku/.lock` (exclusive, non-blocking — fail fast with a clear message if another indexer is running).
- PDF conversion: `ThreadPoolExecutor` (I/O bound).
- Embedding: single-threaded over batches (ORT manages its own intra-op threads).
- SQLite: `PRAGMA journal_mode=WAL` set on every connection.

### 12.6 Crash-safety & resumability
- **Per-file transaction**: each file's chunks + embeddings + `files` row commit in a single `BEGIN ... COMMIT`. Either all of the file's data lands, or none of it does — no half-indexed files.
- **WAL mode** means readers (`haku search`) never block writers (`haku index`), and an OS-level kill during indexing leaves the DB in a consistent state at the last successful per-file commit.
- **Resumability is free**: on the next `haku index` run, the hash check in §12.1 sees all completed files already in `files` and skips them. The interrupted file has no row (transaction rolled back) → it gets retried automatically.
- The `manifest.json` SHA check (see §15) runs on **every** invocation of `index` / `search` before any model load — corrupt or mismatched model files abort the run with a clear error instead of silently producing junk embeddings.

### 12.7 Progress
`tqdm` bar by default. `--quiet` suppresses it (writes nothing to stderr except errors). Useful for cron / scripting.

---

## 13. The ~95-line Chunker (sketch)

**Header policy.** The splitter recognizes only ATX H2 boundaries (`^## `, with a literal space). H1 (`# `) is treated as part of the document body — most personal docs have a single H1 title that we don't want to use as a chunk boundary. H3+ are also ignored to keep boundaries coarse. Setext headings (`---` underlines) and tab-separated forms like `##\t` are not recognized in v1; documents using them will still chunk correctly via the paragraph-level fallback.

**Offset accounting.** `_split_paragraphs` walks the section by re-scanning the string with `find` so multiple consecutive blank lines (which `str.split` would collapse) don't drift the offsets. Offsets are character indices into `source_path` (see §19).

**Overlapping spans.** Two consecutive chunks have **non-overlapping `[start_offset, end_offset]` cores by construction** (the packer assigns each paragraph to exactly one core). The textual overlap visible in `chunks.text` is a *prefix* prepended at index time via `tokenizer.decode(...)` and is not promised to be a literal substring of `source_path`; downstream "jump to source" tools should use the offsets, not search for the overlap text.

```python
# /haku/engine/chunk.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple
from tokenizer import encode, decode

@dataclass
class Chunk:
    chunk_idx:    int       # 0-based, assigned by the chunker
    text:         str
    token_count:  int
    source_path:  str       # threaded through from the caller
    start_offset: int       # char offset into source_path; non-overlapped core
    end_offset:   int

def _split_sections(md: str) -> List[Tuple[int, str]]:
    """Split on H2 (^## ) boundaries. Returns [(abs_char_offset, section_text), ...]."""
    out: List[Tuple[int, str]] = []
    buf: List[str] = []
    buf_start, cur = 0, 0
    for line in md.splitlines(keepends=True):
        if line.startswith("## ") and buf:
            out.append((buf_start, "".join(buf)))
            buf, buf_start = [], cur
        buf.append(line)
        cur += len(line)
    if buf:
        out.append((buf_start, "".join(buf)))
    return out

def _split_paragraphs(section: str, base_offset: int) -> List[Tuple[int, str]]:
    """Blank-line-separated paragraphs, with **accurate offsets** even when
    multiple blank lines appear in a row. We re-scan the string with find()
    rather than relying on split(), which would collapse adjacent separators."""
    out: List[Tuple[int, str]] = []
    sep = "\n\n"
    i, n = 0, len(section)
    while i < n:
        # Skip leading blank-line runs
        while i < n and section.startswith("\n", i):
            i += 1
        if i >= n:
            break
        j = section.find(sep, i)
        end = n if j == -1 else j
        para = section[i:end]
        if para.strip():
            out.append((base_offset + i, para))
        i = end + len(sep) if j != -1 else n
    return out

def chunk_markdown(md: str, source_path: str,
                   max_tokens: int = 512, overlap: int = 64) -> Iterable[Chunk]:
    """Split → pack to max_tokens → apply overlap across boundaries."""
    paras: List[Tuple[int, str]] = []
    for off, section in _split_sections(md):
        paras.extend(_split_paragraphs(section, off))

    # Greedy-pack paragraphs up to max_tokens. Each "core" is a Chunk in
    # the making, without overlap applied yet.
    cores: List[Tuple[int, int, str, int]] = []  # (start, end, text, tokens)
    cur_start, cur_end, cur_text, cur_tokens = None, 0, "", 0
    for off, para in paras:
        n_tok = len(encode(para))
        if cur_start is None:
            cur_start = off
        if cur_tokens + n_tok <= max_tokens:
            cur_text = (cur_text + "\n\n" + para) if cur_text else para
            cur_tokens += n_tok
            cur_end = off + len(para)
        else:
            if cur_text:
                cores.append((cur_start, cur_end, cur_text, cur_tokens))
            cur_start, cur_end, cur_text, cur_tokens = off, off + len(para), para, n_tok
    if cur_text:
        cores.append((cur_start, cur_end, cur_text, cur_tokens))

    # Emit chunks, applying overlap as a tokenized prefix from the previous core.
    for idx, (s, e, text, tokens) in enumerate(cores):
        if idx == 0 or overlap <= 0:
            yield Chunk(idx, text, tokens, source_path, s, e)
            continue
        prev_ids = encode(cores[idx - 1][2])
        tail_ids = prev_ids[-overlap:]
        tail_text = decode(tail_ids)
        merged_text = tail_text + "\n\n" + text
        yield Chunk(
            chunk_idx=idx,
            text=merged_text,
            token_count=len(tail_ids) + tokens,
            source_path=source_path,
            start_offset=s,           # core start, unchanged by overlap
            end_offset=e,             # core end, unchanged by overlap
        )
```

Caller responsibility: pass `source_path` (either `files.path` for text/markdown, or the cached `markdowns/<hash>.md` for PDF/HTML/DOCX). Insertion into `chunks` uses `Chunk.chunk_idx` directly; `file_id` and `id` are assigned by the storage layer.

---

## 14. Shared Tokenizer Module

```python
# /haku/engine/tokenizer.py
from functools import lru_cache
from pathlib import Path
from tokenizers import Tokenizer

# Path resolution is anchored at __file__, NOT at HAKU_HOME. HAKU_HOME only
# affects the bash wrapper (§8); once Python is running, every haku module
# locates sibling directories (models/, markdowns/, logs/) via __file__.
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
```

Imported by `chunk.py`, `embed.py`, and anywhere else token counting is needed. Single source of truth, no double-loading.

**Path resolution policy** (applies to all engine modules):
- `HAKU_HOME` env var: read **only by the bash wrapper** (§8), to pick which Python interpreter to exec.
- Inside Python: `HAKU_ROOT = Path(__file__).resolve().parent.parent`. Every path under `/haku/` is resolved relative to `HAKU_ROOT`. This means a user-relocated install works without any extra plumbing.

---

## 15. Models — Manifest & Status

Models are **manually placed** in `/haku/models/<name>/`. The shipped manifest at **`/haku/engine/manifest.json`** declares the expected files and the SHA-256 each one must match. The manifest ships *with the code*: adding or upgrading a model requires editing this file (so model swaps go through code review, not silent disk edits).

```json
{
  "embedders": {
    "qwen3-embedding-0.6b": {
      "dir": "qwen3-embedding-0.6b",
      "upstream": "onnx-community/Qwen3-Embedding-0.6B-ONNX",
      "revision": "<40-char HF commit SHA>",
      "embedding_dim": 1024,
      "files": [
        { "name": "model.onnx",     "sha256": "abc...", "size": 307200000 },
        { "name": "tokenizer.json", "sha256": "def...", "size": 11200000 }
      ]
    }
  },
  "rerankers": {
    "bge-reranker-v2-m3": {
      "dir": "bge-reranker-v2-m3",
      "upstream": "onnx-community/bge-reranker-v2-m3-ONNX",
      "revision": "<40-char HF commit SHA>",
      "files": [
        { "name": "model.onnx",     "sha256": "ghi...", "size": 284000000 },
        { "name": "tokenizer.json", "sha256": "jkl...", "size": 17100000 }
      ]
    }
  },
  "defaults": {
    "embedder": "qwen3-embedding-0.6b",
    "reranker": "bge-reranker-v2-m3"
  }
}
```

Field semantics:
- `upstream` is the Hugging Face repo path the weights were downloaded
  from. Surfaced verbatim by `haku --version`.
- `revision` is the 40-character HF commit SHA pinning the exact
  upload. Surfaced (truncated) by `haku --version` and in full by
  `haku --version --full`.
- The SHA-256 in `files` is still the source of truth for integrity
  checking. `upstream`/`revision` are for human bug reports and to
  detect "I forgot which upload I trusted last time" mistakes; they
  are **not** consulted at load time.

`--embedder NAME` and `--rerank-model NAME` look up `NAME` in the corresponding section. Unknown name → `haku: unknown model 'X'; declare it in /haku/engine/manifest.json`.

### `haku status` — does NOT load models
Loading models is slow; `status` should answer in ~50ms. It checks:
- Each manifest file exists at its expected path.
- File size matches.
- SHA-256 matches.

Reports `ok` / `missing` / `corrupt` per model. Real loading happens lazily on `index` / `search`.

### Manifest check at load time
Before `index` or `search` instantiates an ONNX session, `embed.py` re-runs the **same** manifest check (existence + size + SHA-256). Why duplicate it?
- A model file could be modified between `haku status` and the actual run.
- A corrupt embedder silently produces garbage vectors that pollute the DB — failing loud at load time prevents this.
- Cost is negligible: SHA-256 of a 600 MB file is ~1–2 s on modern hardware, vs. minutes of indexing work that would otherwise be wasted.

If the check fails at load time, abort with the same `missing`/`corrupt` message format as `status`.

### Status output (text)
```
indexed files:           4,217
db size:                 412.3 MB
schema version:          1
sqlite-vec version:      0.1.9
embedder:                ok  (qwen3-embedding-0.6b)
reranker:                ok  (bge-reranker-v2-m3)
last index started:      2026-05-24T18:42:11Z
last index finished:     2026-05-24T19:03:47Z
last run: indexed/skipped/failed:  37 / 4180 / 7
current run (if active): -
```

Counts come from the `index_runs` table (§19): `last run` = most recent row with `finished_at IS NOT NULL`; `current run` = a row with `finished_at IS NULL` if any. Historical totals across all runs are deliberately **not** surfaced — they're noisy and rarely actionable.

---

## 16. Search Pipeline

```
query → embed → vec_chunks ANN ──┐
                                 ├─→ RRF (k=60) → top-20 → rerank → top-N → output
query → FTS5 BM25 → fts_chunks ──┘                        (optional)
```

### 16.1 Hybrid retrieval
- **Vector side**: cosine similarity over `vec_chunks` (sqlite-vec).
- **Lexical side**: FTS5 BM25 over `fts_chunks` with `tokenize='unicode61 remove_diacritics 2'` (correct Spanish + English handling, folds *café/cafe*, *niño/nino*).
- Fetch top-50 from each side.

### 16.2 RRF fusion
Standard formula, `k = 60`:
```
score(d) = Σ_r  1 / (k + rank_r(d))
```
Take top-20 of the fused list → reranker input.

### 16.3 Re-scoping with `--files`
Applied as a SQL prefix filter **inside both retrieval queries**, before fusion — not as a post-filter. Otherwise top-k gets gutted when the user scopes narrowly.

```sql
-- conceptually:
WHERE files.path LIKE '/home/luis/books/cs/%'
   OR files.path = '/home/luis/papers/foo.pdf'
```

A file qualifies if its stored path **starts with** one of the user-supplied paths (directories) or **equals** it (files). Path normalization (resolve `~`, strip trailing `/`, absolute-ify) happens once at CLI parse time.

**Trailing-slash discipline matters here.** Directory arguments get a trailing `/` re-appended before the `LIKE` pattern is built, so `--files ~/notes` becomes the prefix `/home/luis/notes/%` — not `/home/luis/notes%`. The latter would incorrectly match siblings like `/home/luis/notes-archive/foo.md`. The file-vs-directory distinction is made at parse time via `os.path.isdir`; directories produce a `LIKE` prefix, files produce an `=` equality match.

**Un-indexed scopes**: before retrieval, run `SELECT COUNT(*) FROM files WHERE path LIKE ?` for each `--files` argument. If any returns zero:
```
haku: no indexed files under /home/luis/notes
      run `haku index --files /home/luis/notes` first.
```
Exit non-zero so scripts notice.

### 16.4 Reranking
- **On by default**, `--no-rerank` for speed.
- Model: **`bge-reranker-v2-m3`** (multilingual cross-encoder) via ONNX Runtime, **INT8 dynamic quantization**.
  - Footprint: ~280 MB on disk.
  - Rationale: Qwen3-Reranker is a causal LM used as a pointwise yes/no scorer via templated prompts and `yes`/`no` token logit extraction. ONNX export of that recipe is fragile (tokenizer template drift, logit-index assumptions, hard to verify parity with HF reference). `bge-reranker-v2-m3` is a true cross-encoder with a single forward pass that exports cleanly with `optimum`, supports English + Spanish + 100+ languages, and is a known-good fit for hybrid search rerank stages.
  - Alternative kept in `LICENSES.md` comments: `jina-reranker-v2-multilingual` if BGE licensing or quality ever becomes a blocker.
- Final cut: top-`--top` (default 5).
- **Note**: the original spec called for `Qwen3-Reranker-0.6B`. The swap is documented in the README so users who configured `--rerank-model` against a Qwen export get a clear error pointing at the supported path.

---

## 17. Output Formats

### 17.1 `--format text` (default)

Numbered list, one entry per result, scannable:

```
haku search "turing test citations" --top 3

1. [0.87] /home/luis/books/cs/turing.pdf
   chunk 17 · chars 4821–5333
   ...the imitation game, which we call the Turing test, was proposed
   as a substitute for the question "can machines think?"...

2. [0.81] /home/luis/papers/searle.pdf
   chunk 4 · chars 1280–1742
   ...the Chinese Room argument addresses the Turing test directly,
   claiming that passing it is insufficient for genuine understanding...

3. [0.74] /home/luis/notes/ml.md
   chunk 2 · chars 980–1402
   ...modern critiques argue the Turing test conflates imitation with
   cognition; see Hofstadter (1981) and French (2000)...
```

Bracketed number is the rerank score (or RRF score if `--no-rerank`).

### 17.2 `--format json`

Stable schema, designed for piping. Bump `schema_version` on breaking changes.

```json
{
  "schema_version": 1,
  "query": "turing test citations",
  "took_ms": 142,
  "rerank": true,
  "scoped_paths": ["/home/luis/books/cs"],
  "results": [
    {
      "rank": 1,
      "score": 0.87,
      "score_kind": "rerank",
      "path": "/home/luis/books/cs/turing.pdf",
      "source_path": "/haku/markdowns/a3f1c9...e2.md",
      "chunk_id": 4217,
      "chunk_idx": 17,
      "snippet": "...the imitation game, which we call the Turing test...",
      "start_offset": 4821,
      "end_offset": 5333,
      "token_count": 504
    }
  ]
}
```

Field notes:
- `path` is the original file (what the user wants to open).
- `source_path` is where `start_offset`/`end_offset` apply (same as `path` for text/md; cached `.md` for binary formats).
- `score_kind` is `"rerank"` or `"rrf"` depending on `--no-rerank`.
- `scoped_paths` echoes the `--files` arguments after normalization, or `null` if unscoped.

### 17.3 Use cases for JSON output

The text format is for humans skimming results. JSON exists so other tools can act on them:

```bash
# Open the top result in the system PDF viewer
haku search "turing test" --format json | jq -r '.results[0].path' | xargs xdg-open

# Feed top-5 snippets into a local LLM for a synthesized answer
haku search "what is RRF?" --format json \
  | jq -r '.results[].snippet' \
  | llama-cli -m ~/models/qwen2-7b.gguf -p "Answer from these excerpts:"

# Log searches over time
haku search "$query" --format json --output ~/.haku/history/$(date +%s).json

# Editor integration: Neovim plugin parses JSON, populates quickfix list
```

`--output PATH` writes the same content as stdout but to a file. Useful for long result lists or audit trails.

---

## 18. `haku purge`

**Strict definition**: source file gone from disk → drop the DB row.

**Execution order** (must be preserved — sqlite-vec does not honor `ON DELETE CASCADE`):

1. Acquire the indexer lock (`/haku/.lock`). Concurrent `index` + `purge` would race on the same rows.
2. Find candidates: `SELECT id, path FROM files`. For each row, check `os.path.exists(path)`. Collect `dead_file_ids`.
3. Collect `dead_chunk_ids`: `SELECT id FROM chunks WHERE file_id IN (?, ...)`. **This step happens before any DELETE** — once `files` is touched, the cascade fires and `chunks.id` is gone.
4. `DELETE FROM vec_chunks WHERE rowid IN (?, ...)` using `dead_chunk_ids`.
5. `DELETE FROM files WHERE id IN (?, ...)`. The `ON DELETE CASCADE` from `chunks(file_id)` deletes chunk rows; the `AFTER DELETE` trigger on `chunks` (§19) cleans `fts_chunks` as each row goes.
6. Commit. Release the lock.

Does **not** touch:
- Files whose hash changed (that's `--reindex`'s job).
- Cached markdown in `/haku/markdowns/` (see §21 — orphaned cache files are acceptable v1 disk creep; a `purge --cache` flag is a v2 nice-to-have).

---


# Part IV — Storage & operations

## 19. SQLite Schema

```sql
-- versioning (refuse to run on mismatch)
CREATE TABLE schema_version (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

-- canonical file record.
-- source_path is where chunk offsets refer to:
--   - For .txt/.md/.markdown/.mdx: source_path == path.
--   - For .pdf/.html/.docx: source_path is the cached markdown under /haku/markdowns/.
CREATE TABLE files (
  id          INTEGER PRIMARY KEY,
  path        TEXT NOT NULL UNIQUE,    -- original file on disk (what the user opens)
  source_path TEXT NOT NULL,           -- where chunk offsets refer to
  hash        TEXT NOT NULL,           -- sha256(path||mtime||size)
  mtime       REAL NOT NULL,
  size        INTEGER NOT NULL,
  indexed_at  TEXT NOT NULL
);
CREATE INDEX idx_files_hash ON files(hash);

-- chunk text + positions.
-- start_offset/end_offset are **character offsets** (not bytes) into files.source_path.
-- Offsets refer to the **non-overlapped core** of a chunk (see §13). Two consecutive
-- chunks have non-overlapping core spans by construction; the textual overlap visible
-- in chunks.text is reconstructed at index time and is not promised to be a literal
-- source substring.
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

-- vector index (sqlite-vec)
CREATE VIRTUAL TABLE vec_chunks USING vec0(
  embedding float[1024]
);
-- rowid in vec_chunks == chunks.id (enforced by app code).
-- INVARIANT: vec_chunks.rowid must equal chunks.id for the same chunk, otherwise
-- hybrid search silently joins wrong rows. sqlite-vec is a virtual table and
-- cannot declare a FOREIGN KEY, and we deliberately don't use a trigger
-- (the embedding is computed in Python after the chunk row is inserted).
-- Enforcement is by code contract: storage.py exposes a single helper
-- `insert_chunk_with_embedding(conn, file_id, chunk, embedding)` which is the
-- ONLY supported insertion path. It performs, inside the per-file transaction:
--   1. INSERT INTO chunks(...) RETURNING id      -- get new chunks.id
--   2. INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)  -- same id
-- No other code in the project may write to vec_chunks. Reviewers should reject
-- any direct `INSERT INTO vec_chunks` outside this helper.

-- lexical index (FTS5, content-linked → no text duplication)
CREATE VIRTUAL TABLE fts_chunks USING fts5(
  text,
  content='chunks',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);

-- keep FTS in sync via triggers.
-- For content-linked FTS5, the documented SQLite pattern is AFTER DELETE
-- using the OLD row's values, which are still bound in the trigger context
-- regardless of whether the row has physically left the chunks table. See
-- https://sqlite.org/fts5.html#external_content_tables.
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

-- index-run history (powers `haku status` failed-files count)
CREATE TABLE index_runs (
  run_id         INTEGER PRIMARY KEY,
  started_at     TEXT NOT NULL,
  finished_at    TEXT,                  -- NULL while running
  indexed_count  INTEGER NOT NULL DEFAULT 0,
  skipped_count  INTEGER NOT NULL DEFAULT 0,
  failed_count   INTEGER NOT NULL DEFAULT 0
);
```

### Migrations
- `schema_version` populated on first run.
- On startup, `haku.py` reads `schema_version.version` and compares to the constant `EXPECTED_SCHEMA_VERSION` in `storage.py`.
- **On mismatch: refuse to run.** No automatic migrations in v1 — too easy to corrupt user data silently. Error message:
  ```
  haku: database schema version 1 found, this haku expects version 2.
        back up /haku/database.db if needed, then delete it and re-run `haku index`.
  ```
  Exit non-zero.

### Verifying FTS5 trigger semantics
The triggers above use the documented SQLite FTS5 external-content pattern: `AFTER INSERT` / `AFTER UPDATE` / `AFTER DELETE`, with the delete trigger using `old.id` and `old.text` (both bound regardless of whether the row has physically left). A small SQL smoke test still belongs in CI before locking it in: insert, search, delete, search again — confirm the deleted row no longer matches in `fts_chunks`.

---

## 20. Concurrency Contract (`index` vs `search`)

- **Two `index` runs cannot overlap.** `fcntl.flock` on `/haku/.lock` is exclusive and non-blocking. The second indexer fails fast with:
  ```
  haku: another indexer is running (lock held on /haku/.lock).
  ```
- **`search` does not take the lock.** It runs against the WAL-mode database while indexing is in progress. Results reflect the corpus as of the **last successfully committed file**, which thanks to per-file transactions (§12.6) is always a consistent snapshot.
- **`search` checks the lock and warns** (to stderr only, before printing results):
  ```
  haku: indexing in progress; results reflect the corpus as of the last completed file.
  ```
  This removes the "why are my new files missing" surprise. Suppressed by `--quiet`.
- **`purge` takes the indexer lock** — it mutates `files` and cascades to `chunks`/`vec_chunks`. Concurrent index + purge would race on the same rows.
- **`haku init` must not run concurrently with `index` (or any other haku subcommand).** `init` is a one-shot bootstrap that creates `/haku/.venv` and the database schema; it is **not** lock-protected because the lock file's directory may not exist yet. Running `init` against a haku that's already indexing produces undefined behavior (in the best case, a confusing `pip` failure; in the worst, a half-applied schema rewrite on an open SQLite handle). The contract is: finish `init` before issuing any other subcommand. Re-running `init` after `index` completes is fine and idempotent (§6).

---

## 21. Markdown Cache Lifecycle

Cached PDF/HTML/DOCX→markdown files live in `/haku/markdowns/<sha256-of-source-path>.md`.

- Written **only after successful conversion** — failed conversions leave no cache file.
- Re-used on subsequent `haku index` runs when the source's hash (`path||mtime||size`) is unchanged.
- **Orphaned by source-file rename or deletion**: the cache key is `sha256(source_path)`, not the file's content. If the user renames `~/books/old.pdf` to `~/books/new.pdf`, the old cache entry's key (`sha256("/home/user/books/old.pdf")`) becomes unreachable but stays on disk. SHA-256 collisions on path strings are not a real-world concern; the issue is *unused* entries, not key clashes.
- **Symlinks and bind mounts produce duplicate cache entries.** The cache key hashes the path string as given, not the resolved inode. The same physical PDF reachable at `~/papers/foo.pdf` and `/mnt/usb/papers/foo.pdf` will be converted and cached twice. Not wrong (each path is its own row in `files` too), but worth knowing if you mount the same library in multiple places.
- **`haku purge` does NOT sweep orphaned cache files in v1.** Disk creep is slow (a 200-page PDF produces ~100 KB of markdown). A future `haku purge --cache` flag can add this; tracked as a v2 nice-to-have.

---

## 22. Logging

Everything goes to **JSONL** under `/haku/logs/`. Two files, **one shared base schema** for both:

| Field      | Type   | Notes |
|------------|--------|-------|
| `ts`       | string | ISO-8601 UTC, e.g. `2026-05-24T18:42:11Z` |
| `level`    | string | `info` / `warn` / `error` |
| `event`    | string | machine identifier, e.g. `file_indexed`, `pdf_convert_failed` |
| `run_id`   | integer | matches `index_runs.run_id` (§19); `null` for non-run events |
| `path`     | string | source file path; may be `null` if not applicable |
| `extra`    | object | event-specific fields (chunks, tokens, error, traceback, …) |

### `logs/index.jsonl` — successful operations
```json
{"ts":"2026-05-24T18:42:11Z","level":"info","event":"file_indexed","run_id":42,
 "path":"/x/y.pdf","extra":{"chunks":42,"tokens":18934,"ms":1820}}
```

### `logs/errors.jsonl` — failures (parallel file, same schema)
```json
{"ts":"2026-05-24T18:42:14Z","level":"error","event":"pdf_convert_failed","run_id":42,
 "path":"/x/bad.pdf","extra":{"error":"MuPDF: cannot open","traceback":"..."}}
```

Why JSONL with a uniform schema: trivially grep-able, countable (`wc -l`), parseable (`jq '.extra.error'`), and append-only. Per-run statistics live in the `index_runs` table (§19), not derived from log line counts.

---

## 23. Backup & Restore

**Do not just `cp database.db backup.db`.** WAL mode means writes may be in `database.db-wal` and `database.db-shm` that haven't yet been checkpointed into the main file. A naive copy taken mid-write yields a corrupt backup.

Three safe options, in order of preference:

1. **`VACUUM INTO` (recommended)** — atomic, produces a single fully-checkpointed file:
   ```
   sqlite3 /haku/database.db "VACUUM INTO '/path/to/backup.db';"
   ```
   Safe to run while `haku index` is active.

2. **`sqlite3 .backup`** — the canonical SQLite backup API; also safe during writes:
   ```
   sqlite3 /haku/database.db ".backup '/path/to/backup.db'"
   ```

3. **File copy with checkpoint** — only after explicitly forcing a WAL checkpoint and stopping all writers:
   ```
   sqlite3 /haku/database.db "PRAGMA wal_checkpoint(TRUNCATE);"
   cp /haku/database.db /path/to/backup.db
   ```
   Fragile (depends on no concurrent writer); use options 1 or 2 unless you know why.

**Restore**: stop `haku`, replace `/haku/database.db` with the backup, restart. The schema_version check (§19) will refuse to run if the backup was taken under a different version.

A future `haku backup PATH` subcommand can wrap option 1 with a `--quiet`-respecting progress message; out of scope for v1.

---


# Part V — Planning

## 24. Risks

| Risk | Mitigation |
|------|-----------|
| PyMuPDF4LLM crashes on malformed files | Catch + log to `errors.jsonl`, no DB row → user notified at end of run, retried on next `haku index` (§12.2) |
| sqlite-vec doesn't cascade with `chunks` deletions | Explicit ordered DELETE in purge / reindex paths (§18) |
| FTS5 trigger timing for content-linked tables | Documented SQLite pattern: `AFTER DELETE` using `old.text` (§19); smoke-test in CI |
| User places wrong/corrupt model file in `/haku/models/` | Manifest SHA check in `status` AND at load time (§15) |
| User points `--embedder`/`--rerank-model` at unknown dir | Must be declared in `engine/manifest.json`; unknown name fails fast (§15) |
| Large indexer interrupted mid-run | WAL + per-file transactional commits → safe to resume (§12.6) |
| Stock Python built without sqlite extension loading | `haku init` and `status` smoke-test `vec_version()` and print remediation (§7, §6) |
| `search` running mid-`index` confuses users | Lock check + stderr warning (§20) |
| Orphaned markdown cache after source rename/delete | Documented as acceptable v1 disk creep; `purge --cache` is a v2 (§21) |
| Community-uploaded ONNX quantization differs from upstream FP32 | We trust the `onnx-community/*` upload's quantization (INT8-class). Smoke-test recall on a small EN+ES query set in step 8; if quality regresses materially, swap to a different revision or fall back to a self-exported variant. |
| Community upload silently re-uploaded under same repo | `manifest.json` pins a 40-char `revision` SHA. Re-uploads change the file bytes, the on-disk SHA-256 fails the load-time manifest check (§15), and `haku` aborts with the documented `missing`/`corrupt` message instead of running on unexpected weights. |
| Community upload deleted from HF | New installs cannot fetch the pinned revision. Documented fallback in `MODELS.md` §4: bump to a still-available revision (or a different upload), re-capture hashes, commit the manifest change. |
| Naive `cp` of WAL-mode DB corrupts backup | Documented `VACUUM INTO` / `sqlite3 .backup` procedure (§23) |
| Content edits that preserve `mtime` bypass the incremental skip check | Accepted trade-off in v1: `sha256(path‖mtime‖size)` is cheap but misses rare cases (touch-preserving editors, `tar -p`, `rsync --times`). Users hitting this can force a rebuild with `haku index --reindex`. A content-hash mode is a v2 nice-to-have (§12.1). |
| RAM pressure on 8 GB machines | `--no-rerank` drops peak to ~600 MB (§5) |
| User runs subcommands before bootstrap | `haku init` documented (§6); other subcommands give clear errors if `.venv` or DB are missing |

---

## 25. Build Order

1. `tokenizer.py` + a 10-line test (encode/decode roundtrip).
2. `chunk.py` + golden-file tests for an English doc and a Spanish doc.
3. `storage.py` schema bootstrap + migration refuse-to-run check + per-file transaction helper.
4. `embed.py` embedding path (ONNX Runtime CPU EP, Qwen3-Embedding) + manifest SHA check at load.
5. End-to-end `haku index` on a **mixed test corpus** (with WAL, indexer lock, JSONL logs, tqdm). The corpus must include:
   - 5–6 healthy files spanning every supported extension (`.txt .md .pdf .html .docx`).
   - A zero-byte PDF (PyMuPDF4LLM should fail fast).
   - An encrypted/password-protected PDF (should fail without crashing the run).
   - A mojibake / wrong-encoding `.docx` (handler should log and skip).
   - One Spanish-language doc with accented characters (validates unicode61 + diacritics folding end-to-end).
   These guarantee the skip-and-log path (§12.0, §12.2) is exercised before any of it ends up in production.
6. `haku search` with vector-only retrieval.
7. Add FTS5 + RRF fusion.
8. Wire in `bge-reranker-v2-m3` cross-encoder via ONNX Runtime; A/B against no-rerank on a 20-query smoke set.
9. `haku init`, `haku status`, `haku purge`, `haku --version`, `--quiet`.
10. PDF-failure end-of-run summary, `LICENSES.md`, README.

> IMPORTANT: Each step ends with a working binary. No big-bang integration.
