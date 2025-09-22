import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
_SIM_ONE_PATH = Path(__file__).resolve().parents[1] / "SIM-ONE Training"
if str(_SIM_ONE_PATH) not in sys.path:
    sys.path.append(str(_SIM_ONE_PATH))

from prioritary_mvlm.config import PrioritaryConfig
from prioritary_mvlm.dataset import WeightedTextDataset
from prioritary_mvlm.tokenizer import PrioritaryTokenizer


def test_short_text_is_padded_and_preserves_eos_label(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "sample.txt").write_text("Hello", encoding="utf-8")
    (data_dir / "sample.json").write_text(json.dumps({"id": 1}), encoding="utf-8")

    tokenizer = PrioritaryTokenizer()
    config = PrioritaryConfig(max_length=8, stride=4)
    dataset = WeightedTextDataset(str(data_dir), tokenizer, config)

    assert len(dataset) == 1

    sample = dataset[0]
    input_ids = sample["input_ids"].tolist()
    labels = sample["labels"].tolist()

    assert len(input_ids) == config.max_length

    eos_index = input_ids.index(tokenizer.eos_token_id)
    assert eos_index > 0

    assert all(token == tokenizer.pad_token_id for token in input_ids[eos_index + 1 :])

    assert labels[eos_index - 1] == tokenizer.eos_token_id
    assert labels[eos_index] == tokenizer.pad_token_id
