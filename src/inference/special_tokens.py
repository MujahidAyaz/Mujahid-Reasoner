class SpecialTokenManager:
    def __init__(self, tokenizer) -> None:
        self.pad_token_id = tokenizer.token_to_id("<pad>")
        self.bos_token_id = tokenizer.token_to_id("<bos>")
        self.eos_token_id = tokenizer.token_to_id("<eos>")
        self.unk_token_id = tokenizer.token_to_id("<unk>")

        if self.eos_token_id is None:
            raise ValueError(
                "Tokenizer does not contain an <eos> token."
            )

    def forbidden_token_ids(self) -> tuple[int, ...]:
        token_ids = []

        if self.pad_token_id is not None:
            token_ids.append(self.pad_token_id)

        if self.bos_token_id is not None:
            token_ids.append(self.bos_token_id)

        return tuple(token_ids)
