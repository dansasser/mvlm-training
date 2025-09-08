import json
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .tokenizer import PrioritaryTokenizer
    from .config import PrioritaryConfig


class WeightedTextDataset(Dataset):
    """Dataset class that pairs text files with JSON metadata."""

    def __init__(self, data_dir: str, tokenizer: 'PrioritaryTokenizer', config: 'PrioritaryConfig'):
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.stride = config.stride
        self.examples: List[torch.Tensor] = []
        self.metadata: List[Dict] = []

        data_path = Path(data_dir)
        for txt_file in data_path.rglob("*.txt"):
            json_file = txt_file.with_suffix('.json')
            if not json_file.exists():
                continue
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                self._process_text(text, metadata)
            except Exception:
                continue

    def _process_text(self, text: str, metadata: Dict):
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        for i in range(0, len(tokens) - self.max_length + 1, self.stride):
            example_tokens = tokens[i:i + self.max_length]
            if len(example_tokens) < self.max_length:
                example_tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(example_tokens)))
            self.examples.append(torch.tensor(example_tokens, dtype=torch.long))
            self.metadata.append(metadata)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        tensor = self.examples[idx]
        return {
            'input_ids': tensor,
            'labels': tensor.clone(),
            'metadata': self.metadata[idx]
        }
