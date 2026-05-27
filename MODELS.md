# MODELS.md — acquiring `haku`'s ONNX model files

`haku` needs two ONNX models on disk before step 4 (embedder) and step 8
(reranker). Both are downloaded **pre-quantized from Hugging Face**
community uploads, not exported locally. This is a deliberate trade-off:
faster onboarding (minutes, not 30–60 minutes of CPU export time) at the
cost of trusting a community uploader's quantization choices.

We pin specific Hugging Face commit revisions, so "reproducibility"
means *every install fetches the same bytes*, not *every install
re-derives the same bytes*.

## The two uploads we trust

| Role     | HF repo                                              | File we want                  | Pooling     |
|----------|-------------------------------------------------------|-------------------------------|-------------|
| Embedder | `onnx-community/Qwen3-Embedding-0.6B-ONNX`           | `onnx/model_int8.onnx`        | last-token  |
| Reranker | `onnx-community/bge-reranker-v2-m3-ONNX`             | `onnx/<quantized>.onnx`       | n/a (CE)    |

> ⚠️ **Verify the exact filename in the `onnx/` subdir before downloading.**
> `onnx-community` uploads typically include several variants in the
> `onnx/` directory (`model.onnx`, `model_fp16.onnx`, `model_int8.onnx`,
> `model_quantized.onnx`, `model_uint8.onnx`, etc.). The expected names
> at time of writing are `model_int8.onnx` for the embedder and
> `model_quantized.onnx` for the reranker, but **list the directory
> first** and pick the INT8/quantized variant by inspection — don't
> blind-copy a filename that might have changed.
>
> ⚠️ **Tokenizer location varies between the two repos.** For the
> embedder, `tokenizer.json` lives at the **repo root**. For the
> reranker, `tokenizer.json` is also at the **repo root** (the `onnx/`
> subdir only contains model weights). Both still end up at
> `/haku/models/<name>/tokenizer.json` after copying, so the runtime
> layout matches DESIGN.md §15.

---

## Prereq: `huggingface_hub` CLI

`huggingface_hub` is **not** a `haku` runtime dependency. Install it
in your user-level Python or a throwaway venv — it must not end up in
`/haku/.venv` or `engine/requirements.txt`.

```bash
pip install --user "huggingface_hub[cli]"
# or, in a scratch venv:
#   python3.12 -m venv /tmp/hf && /tmp/hf/bin/pip install "huggingface_hub[cli]"
hf --version    # confirm the CLI is reachable
```

(Recent `huggingface_hub` versions ship the CLI as `hf`. Older releases
still expose `huggingface-cli`; either works.)

---

## 1. Embedder: `onnx-community/Qwen3-Embedding-0.6B-ONNX`

### 1.A Pick a revision

Browse the repo and pin the latest commit on `main`:

```
https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/commits/main
```

Copy the **full 40-char commit SHA** (not the 8-char short form HF
sometimes shows). We'll wire this into `engine/manifest.json` as the
`revision` field — see DESIGN.md §15.

> Pinning to a commit (not `main`) is the whole point: HF uploaders can
> silently re-upload under the same repo name, and we want `haku
> status` to fail loud if that happens.

### 1.B Confirm the exact filenames

Either via the web UI (`Files` tab → `onnx/` subdir) or:

```bash
hf repo files onnx-community/Qwen3-Embedding-0.6B-ONNX --revision <SHA> | grep -E '^(onnx/|tokenizer\.json)'
```

Expected (verify before copying):
- `onnx/model_int8.onnx`
- `tokenizer.json`  (at repo root)

### 1.C Download

```bash
HAKU_ROOT=~/code/haku    # or wherever you cloned haku
REV=<the 40-char commit SHA from step 1.A>

# Download just the two files we need, pinned to the revision.
hf download onnx-community/Qwen3-Embedding-0.6B-ONNX \
    --revision "$REV" \
    --include "onnx/model_int8.onnx" "tokenizer.json" \
    --local-dir /tmp/qwen3-emb-dl
```

### 1.D Lay the files out the way `haku` expects

DESIGN.md §15 expects exactly two files in `/haku/models/qwen3-embedding-0.6b/`:
`model.onnx` and `tokenizer.json`. Rename on copy.

```bash
mkdir -p "$HAKU_ROOT/models/qwen3-embedding-0.6b"

cp /tmp/qwen3-emb-dl/onnx/model_int8.onnx \
   "$HAKU_ROOT/models/qwen3-embedding-0.6b/model.onnx"

cp /tmp/qwen3-emb-dl/tokenizer.json \
   "$HAKU_ROOT/models/qwen3-embedding-0.6b/tokenizer.json"

ls -lh "$HAKU_ROOT/models/qwen3-embedding-0.6b/"
# expect ~300 MB model.onnx, ~11 MB tokenizer.json (DESIGN.md §15 — approximate)
```

### 1.E Capture SHA-256 for `engine/manifest.json`

```bash
cd "$HAKU_ROOT/models/qwen3-embedding-0.6b"
sha256sum model.onnx tokenizer.json
stat -c '%n %s' model.onnx tokenizer.json
```

Paste hashes and byte counts into `engine/manifest.json` under
`embedders.qwen3-embedding-0.6b`. Also fill in the new `upstream`
and `revision` fields per DESIGN.md §15.

---

## 2. Reranker: `onnx-community/bge-reranker-v2-m3-ONNX`

Same workflow as §1.

### 2.A Pick a revision

```
https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/commits/main
```

Copy the full 40-char commit SHA.

### 2.B Confirm the exact filenames

```bash
hf repo files onnx-community/bge-reranker-v2-m3-ONNX --revision <SHA> | grep -E '^(onnx/|tokenizer\.json)'
```

Expected (verify):
- `onnx/model_quantized.onnx`   (this is the INT8 variant `onnx-community` ships)
- `tokenizer.json`  (at repo root)

If you see multiple `onnx/model_*.onnx` files, pick the one whose name
contains `quantized`, `int8`, or `q8`. Skip `fp16` (heavier) and
`uint8` (different output dtype, not what we want).

### 2.C Download

```bash
REV=<the 40-char commit SHA>

hf download onnx-community/bge-reranker-v2-m3-ONNX \
    --revision "$REV" \
    --include "onnx/model_quantized.onnx" "tokenizer.json" \
    --local-dir /tmp/bge-rr-dl
```

### 2.D Lay the files out

```bash
mkdir -p "$HAKU_ROOT/models/bge-reranker-v2-m3"

cp /tmp/bge-rr-dl/onnx/model_quantized.onnx \
   "$HAKU_ROOT/models/bge-reranker-v2-m3/model.onnx"

cp /tmp/bge-rr-dl/tokenizer.json \
   "$HAKU_ROOT/models/bge-reranker-v2-m3/tokenizer.json"

ls -lh "$HAKU_ROOT/models/bge-reranker-v2-m3/"
# expect ~280 MB model.onnx, ~17 MB tokenizer.json (DESIGN.md §15 — approximate)
```

### 2.E Capture SHA-256

```bash
cd "$HAKU_ROOT/models/bge-reranker-v2-m3"
sha256sum model.onnx tokenizer.json
stat -c '%n %s' model.onnx tokenizer.json
```

Paste into `engine/manifest.json` under
`rerankers.bge-reranker-v2-m3`. Fill in `upstream` and `revision`.

---

## 3. Sanity check

After both models are in place:

```bash
cd "$HAKU_ROOT"
./haku status     # should report: embedder ok, reranker ok
```

If `status` says `corrupt` or `missing`, the manifest hashes don't match
the on-disk files — re-copy `sha256sum` output into `engine/manifest.json`
and try again.

---

## 4. When to redo this

- A new revision of the embedder or reranker repo is published and you
  want to upgrade → repeat §1 / §2 with the new SHA, update
  `manifest.json`, note the swap in `NOTES.md`.
- A fresh clone on a new machine → re-run this file's steps. The
  resulting `sha256sum` should match what's committed in
  `manifest.json` (same upstream + same revision = same bytes).
- Hashes don't match what's committed → either the upstream was silently
  re-uploaded (HF allows it) or the download was corrupted. Investigate
  before bumping the committed hashes.

---

## 5. Heads-up: things to verify in step 4 (not now)

Two non-acquisition issues we'll have to deal with when wiring up
`embed.py`:

1. **Pooling strategy.** DESIGN.md §12.4 says "mean-pool + L2-normalize."
   The `onnx-community` model card and the upstream Qwen3 embedding
   recipe both use **last-token pooling** + L2-normalize (Qwen3 is a
   causal LM, not a BERT-style encoder). We'll match the upstream
   recipe in code and patch §12.4 separately. Tracked here so it isn't
   forgotten.

2. **Query prefix.** Qwen3-Embedding expects queries to be wrapped:
   `"Instruct: <task>\nQuery: <text>"`. Documents are embedded raw.
   This asymmetry isn't in DESIGN.md and will need a small addition to
   `embed.py` (a `kind="query"|"document"` argument).

Both are findings, not blockers. Capture them in `NOTES.md` when step 4
opens.
