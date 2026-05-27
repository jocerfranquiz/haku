from __future__ import annotations

import json

import numpy as np
import pytest

from engine.embed import HAKU_ROOT, Embedder, _check_model_files

EMBEDDING_DIM = 1024
MODEL_DIR = HAKU_ROOT / "models" / "qwen3-embedding-0.6b"


# ---------------------------------------------------------------------------
# Manifest integrity checks
# ---------------------------------------------------------------------------


def test_manifest_check_passes_on_valid_files() -> None:
    manifest_path = HAKU_ROOT / "engine" / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)
    spec = manifest["embedders"]["qwen3-embedding-0.6b"]
    _check_model_files(MODEL_DIR, spec["files"])


def test_manifest_check_fails_on_missing_file() -> None:
    specs = [{"name": "nonexistent.onnx", "sha256": "abc", "size": "100"}]
    with pytest.raises(SystemExit, match="model file missing"):
        _check_model_files(MODEL_DIR, specs)


def test_manifest_check_fails_on_size_mismatch() -> None:
    specs = [{"name": "tokenizer.json", "sha256": "abc", "size": "1"}]
    with pytest.raises(SystemExit, match="size mismatch"):
        _check_model_files(MODEL_DIR, specs)


def test_manifest_check_fails_on_sha_mismatch() -> None:
    manifest_path = HAKU_ROOT / "engine" / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)
    tok_spec = manifest["embedders"]["qwen3-embedding-0.6b"]["files"][1]
    bad_spec = {**tok_spec, "sha256": "0" * 64}
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        _check_model_files(MODEL_DIR, [bad_spec])


def test_unknown_model_name_fails() -> None:
    with pytest.raises(SystemExit, match="unknown model 'bogus'"):
        Embedder(name="bogus")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_embed_single_document(embedder: Embedder) -> None:
    vecs = embedder.embed(["Hello world."], kind="document")
    assert len(vecs) == 1
    assert len(vecs[0]) == EMBEDDING_DIM


def test_embed_returns_l2_normalized(embedder: Embedder) -> None:
    vecs = embedder.embed(["The quick brown fox."], kind="document")
    norm = float(np.linalg.norm(vecs[0]))
    assert abs(norm - 1.0) < 1e-5


def test_embed_query_vs_document_differ(embedder: Embedder) -> None:
    text = "semantic search"
    q_vec = embedder.embed([text], kind="query")[0]
    d_vec = embedder.embed([text], kind="document")[0]
    cos_sim = float(np.dot(q_vec, d_vec))
    assert cos_sim < 0.99  # same text, different prefix → different vectors


def test_embed_batch(embedder: Embedder) -> None:
    texts = ["First doc.", "Second doc.", "Third doc."]
    vecs = embedder.embed(texts, kind="document", batch_size=2)
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == EMBEDDING_DIM
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_embed_spanish(embedder: Embedder) -> None:
    vecs = embedder.embed(["El niño comió una manzana en el café."], kind="document")
    assert len(vecs) == 1
    assert len(vecs[0]) == EMBEDDING_DIM


def test_dim_property(embedder: Embedder) -> None:
    assert embedder.dim == EMBEDDING_DIM
