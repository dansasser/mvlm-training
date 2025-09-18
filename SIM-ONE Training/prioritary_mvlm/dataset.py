import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .tokenizer import PrioritaryTokenizer
    from .config import PrioritaryConfig, PropheticSingularityState
else:
    from .config import PropheticSingularityState


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

    def _build_prophetic_state(self, metadata: Dict) -> PropheticSingularityState:
        return PropheticSingularityState.from_metadata(metadata, self.max_length)

    @staticmethod
    def _collate_metadata(batch_metadata: List[Dict]) -> Dict:
        if not batch_metadata:
            return {}

        aggregated: Dict[str, List] = defaultdict(list)
        for sample in batch_metadata:
            for key, value in sample.items():
                aggregated[key].append(value)

        aggregated['__samples__'] = batch_metadata
        return aggregated

    def collate_fn(self, batch: List[Dict]) -> Dict:
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        metadata = [item['metadata'] for item in batch]
        prophetic_states = [item.get('prophetic_state') for item in batch]

        if any(state is None for state in prophetic_states):
            batch_state = PropheticSingularityState.default(
                batch_size=len(batch),
                seq_len=input_ids.size(1),
                device=input_ids.device,
            )
        else:
            batch_state = PropheticSingularityState.batch(prophetic_states)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'metadata': self._collate_metadata(metadata),
            'prophetic_state': batch_state,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        tensor = self.examples[idx]
        return {
            'input_ids': tensor,
            'labels': tensor.clone(),
            'metadata': self.metadata[idx],
            'prophetic_state': self._build_prophetic_state(self.metadata[idx])
        }
