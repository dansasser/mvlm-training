# Prioritary MVLM

## Architecture

`PrioritaryMVLM` builds on the lightweight `SIMONEModel` transformer.
The model pairs with `PrioritaryTokenizer` and can be trained using
`PrioritaryTrainer` for experiments and fine-tuning.

## Training

For a complete walkthrough from environment setup to sampling, see the
[Prioritary Training Complete Guide](PRIORITARY_TRAINING_COMPLETE_GUIDE.md).

1. Prepare a dataset returning token ID tensors.
2. Initialise the tokenizer and model:
   ```python
   from prioritary_mvlm import PrioritaryTokenizer, SIMONEModel
   tokenizer = PrioritaryTokenizer()
   model = SIMONEModel(vocab_size=len(tokenizer))
   ```
3. Use the trainer to run a single training epoch:
   ```python
   from prioritary_mvlm import PrioritaryTrainer
   trainer = PrioritaryTrainer(data_dir="data", output_dir="checkpoints")
   trainer.train()
   ```
