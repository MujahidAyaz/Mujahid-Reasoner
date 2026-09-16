from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.config import ModelConfig, load_model_config


CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


def test_model_config_loads() -> None:
    """Verify the YAML configuration loads correctly."""

    config = load_model_config(CONFIG_PATH)

    assert config.name == "mujahid-reasoner"
    assert config.vocab_size == 32000

    assert config.hidden_size == 256
    assert config.num_layers == 6

    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 4

    assert config.intermediate_size == 768
    assert config.max_sequence_length == 512


def test_head_dimension() -> None:
    """Verify attention head dimension."""

    config = load_model_config(CONFIG_PATH)

    assert config.head_dimension == 32


def test_gqa_group_size() -> None:
    """Verify GQA query-to-KV head grouping."""

    config = load_model_config(CONFIG_PATH)

    assert config.kv_group_size == 2


def test_modern_architecture() -> None:
    """Verify the intended modern architecture components."""

    config = load_model_config(CONFIG_PATH)

    assert config.norm_type == "rmsnorm"
    assert config.activation == "swiglu"
    assert config.position_embedding == "rope"
    assert config.attention_type == "gqa"
    assert config.tie_word_embeddings is True


def test_invalid_attention_configuration() -> None:
    """Verify invalid GQA configurations are rejected."""

    with pytest.raises(ValueError):
        ModelConfig(
            name="invalid",
            vocab_size=32000,
            hidden_size=256,
            num_layers=6,
            num_attention_heads=8,
            num_key_value_heads=3,
            intermediate_size=768,
            max_sequence_length=512,
            dropout=0.0,
            rope_theta=10000.0,
            norm_type="rmsnorm",
            activation="swiglu",
            position_embedding="rope",
            attention_type="gqa",
            tie_word_embeddings=True,
            initializer_range=0.02,
        )


def test_invalid_hidden_size() -> None:
    """Verify hidden size must divide evenly across attention heads."""

    with pytest.raises(ValueError):
        ModelConfig(
            name="invalid",
            vocab_size=32000,
            hidden_size=250,
            num_layers=6,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=768,
            max_sequence_length=512,
            dropout=0.0,
            rope_theta=10000.0,
            norm_type="rmsnorm",
            activation="swiglu",
            position_embedding="rope",
            attention_type="gqa",
            tie_word_embeddings=True,
            initializer_range=0.02,
        )