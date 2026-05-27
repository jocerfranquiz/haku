# LICENSES.md — `haku` dependency and model licenses

## `haku` itself

- **License:** GPL v3
- **Note:** Since `PyMuPDF4LLM` is AGPL v3 and is linked at runtime, the AGPL
  network clause propagates to the combined work. `haku` is a local CLI (no
  network service), so this is fine in practice. See DESIGN.md §2.

## Runtime dependencies

| Package       | Version  | License                     | SPDX                  | Notes |
|---------------|----------|-----------------------------|-----------------------|-------|
| tokenizers    | 0.23.1   | Apache 2.0                  | Apache-2.0            | HuggingFace Rust tokenizer |
| onnxruntime   | 1.26.0   | MIT                         | MIT                   | CPU Execution Provider only |
| sqlite-vec    | 0.1.9    | MIT / Apache 2.0            | MIT OR Apache-2.0     | Loadable SQLite extension |
| pymupdf4llm   | 1.27.2.3 | AGPL v3 (or Artifex commercial) | AGPL-3.0-only    | PDF extraction; AGPL propagates |
| pymupdf       | 1.27.2.3 | AGPL v3 (or Artifex commercial) | AGPL-3.0-only    | Transitive dep of pymupdf4llm |
| selectolax    | 0.4.10   | MIT                         | MIT                   | HTML tag stripping |
| mammoth       | 1.12.0   | BSD 2-Clause                | BSD-2-Clause          | DOCX extraction |
| tqdm          | 4.67.3   | MIT / MPL 2.0               | MPL-2.0 AND MIT       | Progress bar |

## Model weights

| Model                    | Source repo                                        | License      | SPDX         |
|--------------------------|----------------------------------------------------|--------------|--------------|
| Qwen3-Embedding-0.6B    | onnx-community/Qwen3-Embedding-0.6B-ONNX          | Apache 2.0   | Apache-2.0   |
| bge-reranker-v2-m3      | onnx-community/bge-reranker-v2-m3-ONNX            | MIT          | MIT          |

Both model weights are downloaded pre-quantized from `onnx-community/*` Hugging Face
repos and inherit the upstream licenses as derivative works. See `MODELS.md` for
acquisition instructions and pinned revisions.

Users who swap in a different fine-tune or a different ONNX export are responsible
for verifying license compatibility.

## Dev-only dependencies (not shipped)

| Package       | Version  | License      |
|---------------|----------|--------------|
| pytest        | 9.0.3    | MIT          |
| ruff          | 0.15.14  | MIT          |
| mypy          | 2.1.0    | MIT          |
| pre-commit    | 4.6.0    | MIT          |
| python-docx   | 1.2.0    | MIT          |
