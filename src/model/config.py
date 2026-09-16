from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Validated configuration for the Mujahid-Reasoner Transformer."""

    name: str
    vocab_size: int

    hidden_size: int
    num_layers: int

    num_attention_heads: int
    num_key_value_heads: int

    intermediate_size: int

    max_sequence_length: int
    dropout: float

    rope_theta: float

    norm_type: str
    activation: str
    position_embedding: str
    attention_type: str

    tie_word_embeddings: bool

    initializer_range: float

    def __post_init__(self) -> None:
        """Validate architecture invariants."""

        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive."
            )

        if self.hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if self.num_attention_heads <= 0:
            raise ValueError(
                "num_attention_heads must be positive."
            )

        if self.num_key_value_heads <= 0:
            raise ValueError(
                "num_key_value_heads must be positive."
            )

        if (
            self.num_attention_heads
            % self.num_key_value_heads
            != 0
        ):
            raise ValueError(
                "num_attention_heads must be divisible "
                "by num_key_value_heads."
            )

        if (
            self.hidden_size
            % self.num_attention_heads
            != 0
        ):
            raise ValueError(
                "hidden_size must be divisible "
                "by num_attention_heads."
            )

        if self.intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be positive."
            )

        if self.max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be positive."
            )

        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if self.rope_theta <= 0:
            raise ValueError(
                "rope_theta must be positive."
            )

        if self.initializer_range <= 0:
            raise ValueError(
                "initializer_range must be positive."
            )

        if self.norm_type != "rmsnorm":
            raise ValueError(
                "Mujahid-Reasoner currently requires RMSNorm."
            )

        if self.activation != "swiglu":
            raise ValueError(
                "Mujahid-Reasoner currently requires SwiGLU."
            )

        if self.position_embedding != "rope":
            raise ValueError(
                "Mujahid-Reasoner currently requires RoPE."
            )

        if self.attention_type != "gqa":
            raise ValueError(
                "Mujahid-Reasoner currently requires GQA."
            )

    @property
    def head_dimension(self) -> int:
        """Dimension of each attention head."""

        return self.hidden_size // self.num_attention_heads

    @property
    def kv_group_size(self) -> int:
        """Number of query heads sharing each KV head."""

        return (
            self.num_attention_heads
            // self.num_key_value_heads
        )


def load_model_config(
    config_path: Path,
) -> ModelConfig:
    """Load and validate a model configuration from YAML."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model configuration not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError(
            "Model configuration must be a YAML mapping."
        )

    model = raw_config.get("model")
    initialization = raw_config.get(
        "initialization",
        {},
    )

    if not isinstance(model, dict):
        raise ValueError(
            "Missing 'model' configuration."
        )

    if not isinstance(initialization, dict):
        raise ValueError(
            "'initialization' must be a mapping."
        )

    return ModelConfig(
        name=str(model["name"]),
        vocab_size=int(model["vocab_size"]),
        hidden_size=int(model["hidden_size"]),
        num_layers=int(model["num_layers"]),
        num_attention_heads=int(
            model["num_attention_heads"]
        ),
        num_key_value_heads=int(
            model["num_key_value_heads"]
        ),
        intermediate_size=int(
            model["intermediate_size"]
        ),
        max_sequence_length=int(
            model["max_sequence_length"]
        ),
        dropout=float(model["dropout"]),
        rope_theta=float(model["rope_theta"]),
        norm_type=str(model["norm_type"]),
        activation=str(model["activation"]),
        position_embedding=str(
            model["position_embedding"]
        ),
        attention_type=str(
            model["attention_type"]
        ),
        tie_word_embeddings=bool(
            model["tie_word_embeddings"]
        ),
        initializer_range=float(
            initialization["initializer_range"]
        ),
    )