from pathlib import Path

from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"


def load_tokenizer() -> Tokenizer:
    """Load the trained tokenizer."""

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def test_tokenizer_loads() -> None:
    """Verify that the tokenizer can be loaded."""

    tokenizer = load_tokenizer()

    assert tokenizer.get_vocab_size() > 0


def test_english_text() -> None:
    """Verify English text encoding and decoding."""

    tokenizer = load_tokenizer()

    text = "The Transformer architecture is powerful for language modeling."

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)

    assert len(encoded.ids) > 0
    assert decoded.strip() == text


def test_code_text() -> None:
    """Verify programming code is tokenized correctly."""

    tokenizer = load_tokenizer()

    text = 'def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)'

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)

    assert len(encoded.ids) > 0
    assert decoded.strip() == text


def test_mathematics() -> None:
    """Verify mathematical notation is preserved."""

    tokenizer = load_tokenizer()

    text = "Solve: f(x) = x^2 + 2x + 1"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)

    assert len(encoded.ids) > 0
    assert decoded.strip() == text


def test_unicode() -> None:
    """Verify Unicode text can be encoded and decoded."""

    tokenizer = load_tokenizer()

    text = "Pashto: پښتو | Urdu: اردو | café | naïve"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)

    assert len(encoded.ids) > 0
    assert decoded.strip() == text