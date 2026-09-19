def can_stop(
    generated_token_count: int,
    min_new_tokens: int,
) -> bool:
    return generated_token_count >= min_new_tokens


def contains_stop_sequence(
    text: str,
    stop_sequences: tuple[str, ...],
) -> bool:
    return any(
        sequence in text
        for sequence in stop_sequences
    )


def get_safe_stream_text(
    text: str,
    stop_sequences: tuple[str, ...],
) -> tuple[str, bool]:
    if not stop_sequences:
        return text, False

    earliest_stop = None

    for sequence in stop_sequences:
        index = text.find(sequence)

        if index != -1:
            if earliest_stop is None or index < earliest_stop:
                earliest_stop = index

    if earliest_stop is not None:
        return text[:earliest_stop], True

    max_prefix_length = 0

    for sequence in stop_sequences:
        for length in range(1, len(sequence)):
            if text.endswith(sequence[:length]):
                max_prefix_length = max(
                    max_prefix_length,
                    length,
                )

    if max_prefix_length:
        return text[:-max_prefix_length], False

    return text, False
