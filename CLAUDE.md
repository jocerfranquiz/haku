# CLAUDE.md — `haku` implementation guide

You are helping me implement `haku`, a local, CPU-only, multilingual (English +
Spanish) semantic + lexical hybrid search CLI. The full design lives in
`DESIGN.md` (the design document I gave you). **Read it before doing anything.**
It is the single source of truth — when in doubt, re-read the relevant section
rather than guessing.

This file governs *how we work together*, not what we're building.

---

## Repo map (what each file is for)

| File / dir         | Role                                                                 |
|--------------------|----------------------------------------------------------------------|
| `DESIGN.md`        | Authoritative spec. Cite section numbers when you reference it.      |
| `CLAUDE.md`        | This file. Workflow rules for you.                                   |
| `MODELS.md`        | How the ONNX model files are acquired (HF community pre-quantized, pinned by revision). Read before step 4 / 8. |
| `NOTES.md`         | My learning log + your end-of-step session log (see "Session log").  |
| `glossary.md`      | Terms I'm learning, defined in my own words as they come up.         |
| `Makefile`         | `make test`, `make lint`, etc. Use it in your test checklists.       |
| `.python-version`  | Pinned to 3.12. Do not switch interpreters without asking.           |
| `.gitignore`       | Already in place; respect it.                                        |
| `engine/`          | All Python source. Layout per DESIGN.md §4.                          |
| `engine/manifest.json` | Model hashes + upstream + revision. Populated by me from `MODELS.md`, not by you. |

---

## Prime directives

1. **DESIGN.md is authoritative.** If something in this file appears to
   contradict DESIGN.md, DESIGN.md wins. Tell me about the contradiction.
2. **Incremental, not big-bang.** We follow the build order in §25 of
   DESIGN.md, one step at a time. Do not jump ahead.
3. **I am learning.** After each step you stop and let me test, read, and
   ask questions. Do not start the next step on your own.
4. **No scope creep.** Stick to what the current step requires. Future
   steps will pull in what they need when they need it.

---

## Model strategy (one-liner so you don't go looking)

The two ONNX models are **downloaded pre-quantized from Hugging Face**
community uploads (`onnx-community/Qwen3-Embedding-0.6B-ONNX` and
`onnx-community/bge-reranker-v2-m3-ONNX`), pinned to specific commit
revisions. We do **not** export or quantize anything locally. Full
procedure: `MODELS.md`.

Consequences for you:

- `engine/manifest.json` gains two fields per model entry: `upstream`
  (the HF repo string, e.g. `onnx-community/Qwen3-Embedding-0.6B-ONNX`)
  and `revision` (the 40-char commit SHA). I fill these in; you read
  them in `embed.py` and surface them via `haku --version`.
- The license note in DESIGN.md §2 refers to the upstream model
  licenses (Apache 2.0 / MIT). The ONNX-converted derivative inherits
  those licenses on `onnx-community`; do not invent extra license
  strings.
- `huggingface_hub` is **not** a runtime dep. Do not add it to
  `engine/requirements.txt`. It only lives in `MODELS.md`'s install
  instructions for me.

---

## How we work, step by step

The build order has 10 steps (DESIGN.md §25). For **every** step:

### Before writing any code
- Re-read the relevant DESIGN.md sections for this step.
- Tell me, in 3–6 lines:
  - what files you will create or modify,
  - which DESIGN.md sections apply,
  - any decisions or assumptions you are about to make,
  - anything in the design you find ambiguous.
- **Wait for me to say "go"** before writing code. If I correct an
  assumption, fold the correction in before starting.

### While implementing
- Match DESIGN.md line for line on names, paths, schemas, flags, error
  messages, and licensing notes. The doc is opinionated on purpose.
- Keep modules small. The chunker is ~95 lines for a reason; do not let
  files balloon.
- Comments should explain *why*, not *what*. Reference DESIGN.md sections
  inline (e.g. `# see §12.6 — per-file transaction`).
- No new runtime dependencies beyond what DESIGN.md lists for the current
  step. If you think one is needed, stop and ask.
- Python **3.12** (pinned in `.python-version`). Type hints on every
  public function. `from __future__ import annotations` at the top of
  each engine module.
- No `print()` for status — use the logging path defined in §22 once it
  exists, and stderr otherwise. Respect `--quiet` whenever it's in scope.
- Code must pass `make check` (ruff + mypy + pytest). Configure ruff and
  mypy in step 1 alongside the tokenizer. See "Tooling" below.

### After implementing
- Run `make check` and paste the output.
- Produce a short "what to test" checklist for me:
  - 1–3 commands to run (prefer `make` targets when possible),
  - what success looks like,
  - what known limitations exist at this step (e.g. "no FTS yet, vector-only").
- **Append a session-log entry to `NOTES.md`** (see "Session log" below).
- **Stop. Do not start the next step.** Wait for me to say "next" (or
  give corrections). If I ask questions, answer them without editing code
  unless I explicitly ask for a change.

---

## The 10 steps (recap from DESIGN.md §25)

Each step ends with a **working binary** I can run. Do not collapse steps.

| # | Deliverable                                                     | Binary state after step                              |
|---|------------------------------------------------------------------|------------------------------------------------------|
| 1 | `tokenizer.py` + encode/decode roundtrip test + ruff/mypy setup | `make check` passes; tokenizer roundtrips           |
| 2 | `chunk.py` + golden-file tests (EN + ES)                        | `make test` green on both fixtures                   |
| 3 | `storage.py` schema bootstrap + version-mismatch refuse + tx helper | Can create DB, refuse on bad version, insert chunk+vec atomically |
| 4 | `embed.py` ONNX path + manifest SHA check at load               | Can embed a string, fails loud on bad model files    |
| 5 | End-to-end `haku index` on mixed corpus (lock, WAL, JSONL, tqdm)| `haku index --files <corpus>` runs to completion, error fixtures land in `errors.jsonl` |
| 6 | `haku search` — vector-only                                     | `haku search "query"` returns ranked hits            |
| 7 | + FTS5 + RRF fusion                                             | Same command, now hybrid                             |
| 8 | + `bge-reranker-v2-m3` cross-encoder, A/B against `--no-rerank` | Reranked results by default, opt-out works          |
| 9 | `haku init`, `status`, `purge`, `--version`, `--quiet`          | Full CLI surface from §9 works                       |
| 10| PDF-failure end-of-run summary, `LICENSES.md`, README           | Project shippable                                    |

For step 5's test corpus, you generate the fixtures yourself (a Python
script that produces the healthy files, the zero-byte PDF, the encrypted
PDF, the mojibake docx, the Spanish-accented doc). Do not download
copyrighted material.

For steps **4 and 8**, the model files must already be on disk per
`MODELS.md`. If `/haku/models/<name>/` is empty, stop and tell me — do
not stub or mock the model.

---

## Tooling (set up in step 1, used every step after)

Step 1 is more than just `tokenizer.py`. It also establishes the
quality-gate tooling we run every subsequent step. Specifically, step 1
must:

1. Create `engine/requirements.txt` with strict `==` pins (DESIGN.md §6).
   For step 1 this is just `tokenizers==<latest>` plus dev deps.
2. Create `engine/requirements-dev.txt` with `pytest`, `ruff`, `mypy`
   (also `==`-pinned).
3. Add a `pyproject.toml` (or `ruff.toml` + `mypy.ini` — your call, pick
   one and be consistent) with **strict-ish** settings:
   - ruff: enable at minimum `E,F,W,I,UP,B,SIM,RET,N,PL` rule families;
     line length 100.
   - mypy: `strict = True`, `python_version = "3.12"`,
     `warn_unused_ignores = True`. We accept that this is painful for
     SQLite/ONNX boundaries — handle untyped externals with explicit
     `# type: ignore[reason]` only where unavoidable.
4. Confirm `make check` runs and passes.
5. Set up a `.pre-commit-config.yaml` that runs `ruff check`,
   `ruff format --check`, and `pytest -x` on commit. Document the
   `pre-commit install` command in the step's "what to test"
   checklist so I can install the hook on my side.

Add new deps **only** in the step that introduces them
(e.g. `onnxruntime` in step 4, `pymupdf4llm` in step 5, `sqlite-vec` in
step 3). Do not pre-install future deps.

---

## Session log (end-of-step entries in `NOTES.md`)

At the end of every step, **append** a block to `NOTES.md` in this exact
format. No edits to my existing notes — only append. Keep it ~5 lines.

```markdown
## Step N — <short title>
- **Date:** YYYY-MM-DD
- **Shipped:** what now works (1 line)
- **Surprised me:** something I didn't expect or that diverged from the design (1 line, or "nothing")
- **Deferred:** anything I noticed but punted to a later step (1 line, or "nothing")
- **Open questions for Luis:** anything you'd like me to think about before "next" (1 line, or "none")
```

This is your handoff note to me and to your future self when the next
session opens with no memory of this one. Be honest — "surprised me:
nothing" is fine if it's true.

---

## Conventions to keep us aligned

- **Project root**: `/haku` per DESIGN.md, but I may run it under
  `~/code/haku` via `HAKU_HOME`. Path resolution inside Python uses
  `HAKU_ROOT = Path(__file__).resolve().parent.parent` (DESIGN.md §14).
  `HAKU_HOME` is read **only** by the bash wrapper.
- **Repo layout** must match DESIGN.md §4 exactly. Do not add directories
  not listed there without telling me first.
- **`requirements.txt`** uses strict `==` pins on every line (§6
  idempotency). Add deps incrementally as steps require them; do not
  pre-pin things we don't use yet.
- **Error messages** quoted in DESIGN.md are *exact* — copy them
  verbatim (sqlite-vec load failure §7, schema mismatch §19, unknown
  model §15, indexer lock held §20, un-indexed scope §16.3, etc.).
- **License hygiene**: `haku` is GPL v3. `PyMuPDF4LLM` is AGPL v3 — note
  this anywhere it gets imported. Model weights tracked in
  `LICENSES.md` (step 10). The ONNX-converted weights inherit the
  upstream license (Apache 2.0 for Qwen3-Embedding, MIT for bge-reranker).
- **Testing**: pytest. Tests live next to code (`engine/test_*.py`) and
  must run from a stock venv with only `requirements.txt` +
  `requirements-dev.txt` installed. No network calls in tests.
- **Git**: one commit per step, message format
  `step N: <summary>` (e.g. `step 2: chunk.py + EN/ES golden tests`).
  Do not commit on my behalf — show me the diff and the proposed
  message, I'll commit.
- **Glossary**: `glossary.md` lists terms I want to define in my own
  words as they come up. You may *suggest* I add a term, but **never
  write the definition for me** — that defeats the learning goal.

---

## Things you should not do without asking

- Skip ahead to a later step, even if "it's only 5 lines."
- Add a runtime dependency not already in DESIGN.md.
- Introduce a new subcommand, flag, table, column, or log field.
- Change a DESIGN.md-quoted error message, even to "improve" it.
- Use a model other than `Qwen3-Embedding-0.6B` / `bge-reranker-v2-m3`,
  or a source other than the `onnx-community/*` repos pinned in
  `MODELS.md`.
- Touch `init`'s behavior beyond what §6 spells out (no `--force`).
- Add automatic schema migrations (§19 is explicit: refuse to run).
- Reach for GPU, async, multiprocessing-for-embedding, or a daemon.
- Use `print` for progress instead of `tqdm` / stderr.
- Fill in `engine/manifest.json` hashes, `upstream`, or `revision`
  yourself — those come from my local `MODELS.md` run, not from your
  imagination.
- Add `huggingface_hub` to runtime deps. It's an acquisition tool, not
  a `haku` dependency.
- Write definitions in `glossary.md`. That file is mine.

If you think one of these is genuinely warranted, **stop and ask** with
a one-paragraph case. I'd rather have the conversation than the surprise.

---

## What to do if you get stuck

1. Re-read the relevant DESIGN.md section.
2. State the ambiguity in one sentence.
3. Propose 2 options with trade-offs.
4. Stop and wait for me. Do not pick for me silently.

---

## What "done" looks like for a step

Done means **all** of these:

- Code matches DESIGN.md for names, paths, schemas, messages.
- `make check` passes locally.
- I can run a command and see the expected behavior.
- Known limitations at this step are written down.
- The session-log entry has been appended to `NOTES.md`.
- You have stopped and handed control back to me.

Then, and only then, we move to the next step.
