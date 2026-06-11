"""Tests for src/embeddings.py | EmbeddingRegistry and model config."""

import pytest
from src.embeddings import EMBEDDING_MODELS, EmbeddingRegistry


class TestEmbeddingModels:
    """Verify EMBEDDING_MODELS registry structure."""

    def test_has_minilm(self):
        assert "minilm" in EMBEDDING_MODELS

    def test_has_bge(self):
        assert "bge" in EMBEDDING_MODELS

    def test_model_has_required_keys(self):
        for key, cfg in EMBEDDING_MODELS.items():
            assert "name" in cfg, f"{key} missing 'name'"
            assert "dim" in cfg, f"{key} missing 'dim'"

    def test_minilm_dim_384(self):
        assert EMBEDDING_MODELS["minilm"]["dim"] == 384

    def test_bge_dim_768(self):
        assert EMBEDDING_MODELS["bge"]["dim"] == 768

    def test_bge_has_query_prefix(self):
        assert "query_prefix" in EMBEDDING_MODELS["bge"]
        assert EMBEDDING_MODELS["bge"]["query_prefix"] != ""

    def test_minilm_has_name(self):
        assert EMBEDDING_MODELS["minilm"]["name"] == "all-MiniLM-L6-v2"


class TestEmbeddingRegistry:
    """Test EmbeddingRegistry class (structure only, no model loading)."""

    def test_init_empty(self):
        reg = EmbeddingRegistry()
        assert reg.loaded_models == []

    def test_invalid_key_raises(self):
        reg = EmbeddingRegistry()
        with pytest.raises(ValueError, match="Unknown model key"):
            reg.load_model("nonexistent")

    def test_encode_without_load_raises(self):
        reg = EmbeddingRegistry()
        with pytest.raises(ValueError, match="not loaded"):
            reg.encode("minilm", ["test"])

    def test_get_dim_returns_int(self):
        reg = EmbeddingRegistry()
        assert reg.get_dim("minilm") == 384
        assert reg.get_dim("bge") == 768
