"""ONNX Runtime embedder with manifest integrity check. See DESIGN.md §12.4, §15."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from tokenizers import Tokenizer

from engine.tokenizer import get_tokenizer

if TYPE_CHECKING:
    from numpy.typing import NDArray

HAKU_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = HAKU_ROOT / "engine" / "manifest.json"

# see §12.4 — upstream Qwen3-Embedding query instruction
_DEFAULT_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _load_manifest() -> dict[str, object]:
    with MANIFEST_PATH.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def _check_model_files(model_dir: Path, file_specs: list[dict[str, str]]) -> None:
    """Verify existence, size, and SHA-256 for each file in the manifest. See §15."""
    for spec in file_specs:
        fpath = model_dir / spec["name"]
        if not fpath.exists():
            msg = f"haku: model file missing: {fpath}"
            raise SystemExit(msg)

        actual_size = fpath.stat().st_size
        expected_size = int(spec["size"])
        if actual_size != expected_size:
            msg = (
                f"haku: model file corrupt (size mismatch): {fpath}\n"
                f"      expected {expected_size} bytes, got {actual_size}"
            )
            raise SystemExit(msg)

        h = hashlib.sha256()
        with fpath.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != spec["sha256"]:
            msg = f"haku: model file corrupt (SHA-256 mismatch): {fpath}"
            raise SystemExit(msg)


def _resolve_model(
    section: str, name: str | None, default_key: str,
) -> dict[str, object]:
    """Look up a model in the manifest by section. See §15."""
    manifest = _load_manifest()
    models: dict[str, object] = manifest[section]  # type: ignore[assignment]
    if name is None:
        name = manifest["defaults"][default_key]  # type: ignore[index]
    if name not in models:
        msg = f"haku: unknown model '{name}'; declare it in /haku/engine/manifest.json"
        raise SystemExit(msg)
    return models[name]  # type: ignore[return-value]


def _resolve_embedder(name: str | None = None) -> dict[str, object]:
    return _resolve_model("embedders", name, "embedder")


def _resolve_reranker(name: str | None = None) -> dict[str, object]:
    return _resolve_model("rerankers", name, "reranker")


# Qwen3-0.6B architecture: 28 layers, 8 KV heads, 128 head_dim
_NUM_LAYERS = 28
_NUM_KV_HEADS = 8
_HEAD_DIM = 128


def _build_session(model_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0  # see §10 — let ORT choose
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])


def _last_token_pool(
    hidden_states: NDArray[np.float32],
    attention_mask: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Extract the last non-padding token's hidden state. See §12.4."""
    seq_lengths = attention_mask.sum(axis=1)
    last_indices = seq_lengths - 1
    batch_indices = np.arange(hidden_states.shape[0])
    return hidden_states[batch_indices, last_indices]  # type: ignore[no-any-return]


def _l2_normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms  # type: ignore[no-any-return]


class Embedder:
    """Lazy-loading ONNX embedder with manifest integrity check."""

    def __init__(self, name: str | None = None) -> None:
        self._spec = _resolve_embedder(name)
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._dim: int = self._spec["embedding_dim"]  # type: ignore[assignment]

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        model_dir = HAKU_ROOT / "models" / str(self._spec["dir"])
        _check_model_files(model_dir, self._spec["files"])  # type: ignore[arg-type]
        self._session = _build_session(model_dir / "model.onnx")
        self._tokenizer = get_tokenizer()

    def _prepare_inputs(
        self, texts: list[str],
    ) -> dict[str, NDArray[np.int64] | NDArray[np.float32]]:
        """Tokenize a batch and build the full ONNX input feed including
        position_ids and empty KV cache tensors."""
        assert self._tokenizer is not None
        self._tokenizer.enable_padding()
        encodings = self._tokenizer.encode_batch(texts)
        self._tokenizer.no_padding()

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encodings], dtype=np.int64,
        )
        batch, seq_len = input_ids.shape
        position_ids = np.broadcast_to(
            np.arange(seq_len, dtype=np.int64), (batch, seq_len),
        ).copy()

        feed: dict[str, NDArray[np.int64] | NDArray[np.float32]] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        # Empty KV cache: past_sequence_length = 0
        empty_kv = np.zeros(
            (batch, _NUM_KV_HEADS, 0, _HEAD_DIM), dtype=np.float32,
        )
        for layer in range(_NUM_LAYERS):
            feed[f"past_key_values.{layer}.key"] = empty_kv
            feed[f"past_key_values.{layer}.value"] = empty_kv
        return feed

    def embed(
        self,
        texts: list[str],
        kind: Literal["query", "document"] = "document",
        batch_size: int = 16,
    ) -> list[list[float]]:
        """Embed a list of texts. See §12.4 for query vs document asymmetry."""
        self._ensure_loaded()
        assert self._session is not None

        if kind == "query":
            texts = [
                f"Instruct: {_DEFAULT_TASK}\nQuery: {t}" for t in texts
            ]

        all_embeddings: list[NDArray[np.float32]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            feed = self._prepare_inputs(batch)
            outputs = self._session.run(
                ["last_hidden_state"], feed,
            )
            hidden_states = outputs[0]
            attention_mask: NDArray[np.int64] = feed["attention_mask"]  # type: ignore[assignment]
            pooled = _last_token_pool(hidden_states, attention_mask)
            normed = _l2_normalize(pooled)
            all_embeddings.append(normed)

        result = np.concatenate(all_embeddings, axis=0)
        return result.tolist()  # type: ignore[no-any-return]


class Reranker:
    """Lazy-loading ONNX cross-encoder reranker. See §16.4."""

    def __init__(self, name: str | None = None) -> None:
        self._spec = _resolve_reranker(name)
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        model_dir = HAKU_ROOT / "models" / str(self._spec["dir"])
        _check_model_files(model_dir, self._spec["files"])  # type: ignore[arg-type]
        self._session = _build_session(model_dir / "model.onnx")
        tok_path = model_dir / "tokenizer.json"
        self._tokenizer = Tokenizer.from_file(str(tok_path))

    def score(
        self, query: str, passages: list[str], batch_size: int = 16,
    ) -> list[float]:
        """Score (query, passage) pairs. Returns one logit per passage."""
        self._ensure_loaded()
        assert self._session is not None
        assert self._tokenizer is not None

        all_scores: list[float] = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            self._tokenizer.enable_padding()
            self._tokenizer.enable_truncation(max_length=512)
            encodings = self._tokenizer.encode_batch(
                [(query, p) for p in batch],
            )
            self._tokenizer.no_padding()
            self._tokenizer.no_truncation()

            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64,
            )
            outputs = self._session.run(
                None, {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            logits = outputs[0].flatten().tolist()
            all_scores.extend(logits)

        return all_scores
