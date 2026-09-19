from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    min_new_tokens: int = 0
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: tuple[str, ...] = ()