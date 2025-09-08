class PrioritaryTokenizer:
    """Simple character-level tokenizer for SIM-ONE."""

    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.offset = 3
        self.vocab_size = 128 + self.offset

    def encode(self, text, add_special_tokens=True):
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for ch in text:
            ids.append(ord(ch) % 128 + self.offset)
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            if i >= self.offset:
                chars.append(chr((i - self.offset) % 128))
        return ''.join(chars)

    def __len__(self):
        return self.vocab_size
