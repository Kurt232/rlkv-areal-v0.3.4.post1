# Mixed Attention Training

This document describes the "Mixed Attention Training" feature, a parameter-efficient
fine-tuning (PEFT) method implemented in AReaL. It is designed to efficiently adapt
Large Language Models (LLMs) for long-context reasoning tasks.

## Overview

Mixed Attention Training combines two types of attention mechanisms:

1. **Standard Dense Attention:** Implemented using the highly optimized `flash-attn`.
1. **Block-Sparse Streaming Attention:** A memory-efficient attention mechanism suitable
   for handling long sequences, based on a sink window and a sliding local window.

A small, trainable **adapter** parameter is introduced in each attention layer. This
adapter learns a dynamic weighting to interpolate between the outputs of the dense and
sparse attention mechanisms for each attention head. This allows the model to learn how
much to rely on local (streaming) versus global (dense) context.

The core idea is to fine-tune only these small adapter weights instead of the entire
model, making the process significantly more memory and computationally efficient.

## How It Works

When Mixed Attention Training is enabled:

1. The standard attention mechanism of a supported LLM is replaced with a custom
   `mixed_attention_forward` function.
1. A new trainable `adapter` parameter is added to each attention block.
1. During the forward pass, both attention scores are calculated. The final attention
   output is a learned combination of the two:
   ```
   output = adapter * dense_attention_output + (1.0 - adapter) * streaming_attention_output
   ```
1. Only the `adapter` parameters are marked as trainable (`requires_grad=True`). All
   other parameters of the pre-trained model are frozen.

## How to Use

To enable this feature for a training run, simply set the following flag in your
experiment configuration YAML file:

```yaml
# In your config.yaml
actor:
  ...
  enable_mixed_attn_training: true
  # You can also configure the streaming attention window sizes
  sink_window_size: 128
  recent_window_size: 1024
```

When `enable_mixed_attn_training` is `true`, the `FSDPEngine` will automatically handle
the model modification and freeze the appropriate weights before training begins.

## Checkpointing (Save & Load)

The checkpointing process is designed to be efficient and only store the trained adapter
weights.

### Saving

When you save a model trained with mixed attention:

- The script extracts the `adapter` weights from the model's state dictionary.
- These weights are saved into a JSON file named `adapter_weight.json` within your
  specified checkpoint directory.
- The original, frozen weights of the base LLM are **not** saved, saving significant
  disk space.

### Loading

To resume training or for inference:

- The `FSDPEngine` first loads the original, pre-trained base model (e.g., from Hugging
  Face Hub).
- It then loads the `adapter` weights from the `adapter_weight.json` file in the
  checkpoint directory.
- These adapter weights are injected back into the corresponding attention layers of the
  model.

This ensures that you have the fully constituted, fine-tuned model in memory for further
use.

## Supported Models

This feature is currently implemented for the following model architectures:

- Llama
- Phi-3
- Qwen2
