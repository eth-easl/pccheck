---
license: other
tags:
- generated_from_trainer
datasets:
- wikitext
model-index:
- name: output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output

This model is a fine-tuned version of [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) on the wikitext wikitext-2-raw-v1 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.31.0.dev0
- Pytorch 2.1.2+cu121
- Datasets 3.1.1.dev0
- Tokenizers 0.13.3
