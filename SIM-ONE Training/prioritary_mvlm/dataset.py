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
        self.sequence_lengths: List[int] = []

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
        if not tokens:
            return

        pad_token_id = self.tokenizer.pad_token_id

        def append_window(window_tokens: List[int]) -> None:
            tokens_list = list(window_tokens)
            if not tokens_list:
                return

            sequence_length = min(len(tokens_list), self.max_length)
            tokens_list = tokens_list[:self.max_length]

            if sequence_length < self.max_length:
                tokens_list.extend([pad_token_id] * (self.max_length - sequence_length))

            self.examples.append(torch.tensor(tokens_list, dtype=torch.long))
            self.metadata.append(metadata)
            self.sequence_lengths.append(sequence_length)

        if len(tokens) <= self.max_length:
            append_window(tokens)
            return

        last_start = None
        limit = len(tokens) - self.max_length + 1
        for start in range(0, limit, self.stride):
            append_window(tokens[start:start + self.max_length])
            last_start = start

        final_start = max(len(tokens) - self.max_length, 0)
        if last_start is None or final_start > last_start:
            append_window(tokens[final_start:])

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
        seq_len = self.sequence_lengths[idx] if idx < len(self.sequence_lengths) else tensor.size(0)
        labels = tensor.clone()

        if labels.size(0) > 1:
            labels[:-1] = tensor[1:]

        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)

        pad_value = -100
        if pad_token_id is not None and pad_token_id != eos_token_id:
            pad_value = pad_token_id

        if seq_len <= 1:
            labels.fill_(pad_value)
        else:
            end_index = min(seq_len - 1, labels.size(0))
            labels[end_index:] = pad_value

        return {
            'input_ids': tensor,
            'labels': labels,
            'metadata': self.metadata[idx],
            'prophetic_state': self._build_prophetic_state(self.metadata[idx])
        }
