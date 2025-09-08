# Prioritary Training Complete Guide

This guide walks through configuring your environment, preparing a priority-aware dataset and training the `PrioritaryMVLM` model. It mirrors the lightweight CLI shipped with this repository and highlights how example priorities influence learning.

## Environment setup

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -e .
   ```

## Dataset preparation

Training uses pairs of text files and JSON metadata:

```
my_corpus/
├── sample.txt
└── sample.json
```

The JSON may contain a `priority` (or `score`) field. Higher values weight the sample more heavily during training.

```json
{
  "source": "sermon",
  "priority": 2.0
}
```

## Running the CLI

Invoke the trainer through the CLI in the `cli/` directory:

```bash
python cli/train_prioritary.py \
    --train-data my_corpus \
    --output-dir runs/exp1 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --epochs 3
```

Use `--val-data` to supply a validation set. Priority weights from the metadata are averaged per batch and scale the loss, emphasising important examples.

### Customising configuration

For finer control create a `PrioritaryConfig` and pass it to the trainer:

```python
from prioritary_mvlm.config import PrioritaryConfig
from prioritary_mvlm.trainer import PrioritaryTrainer
cfg = PrioritaryConfig(num_epochs=5, batch_size=4, max_length=256)
trainer = PrioritaryTrainer(data_dir="my_corpus", output_dir="runs/custom", config=cfg)
trainer.train()
```

## Checkpointing and resuming

Checkpoints are written to the `output_dir` at evaluation intervals. Resume training with:

```bash
python cli/train_prioritary.py \
    --train-data my_corpus \
    --output-dir runs/exp1 \
    --resume-from runs/exp1/checkpoint_step100.pt
```

Or programmatically:

```python
trainer = PrioritaryTrainer(data_dir="my_corpus", output_dir="runs/exp1")
trainer.load_checkpoint("runs/exp1/checkpoint_step100.pt")
trainer.train()
```

## Evaluation

The trainer periodically evaluates on the training set or an optional validation set. Run a manual evaluation with:

```python
loss, ppl = trainer.evaluate()
print(f"eval_loss={loss:.4f} perplexity={ppl:.2f}")
```

## Sample generation

After training or loading a checkpoint you can generate text samples:

```python
text = trainer.generate_sample("In the beginning", max_length=50)
print(text)
```

This is useful for quick qualitative checks on model behaviour.
