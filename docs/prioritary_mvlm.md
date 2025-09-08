# Prioritary MVLM

## Architecture

`PrioritaryMVLM` wraps a minimal GPT-2 language model provided by the
`transformers` library.  The model can be paired with `PrioritaryTokenizer`
and trained using `PrioritaryTrainer` for experiments and fine-tuning.

## Training

1. Prepare a dataset returning token ID tensors.
2. Initialise the tokenizer and model:
   ```python
   from prioritary_mvlm import PrioritaryTokenizer, PrioritaryMVLM
   tokenizer = PrioritaryTokenizer.from_pretrained("gpt2")
   model = PrioritaryMVLM()
   ```
3. Use the trainer to run a single training epoch:
   ```python
   from prioritary_mvlm import PrioritaryTrainer
   trainer = PrioritaryTrainer(model, dataset)
   trainer.train_epoch()
   ```
