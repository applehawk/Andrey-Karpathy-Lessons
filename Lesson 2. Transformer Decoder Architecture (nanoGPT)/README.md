# Lesson 2: Transformer Decoder Architecture (nanoGPT)

In this lesson, we explore the internal architecture of a Transformer Decoder (GPT-style). We focus on understanding how Multi-Head Attention works, how residual blocks are structured, and the importance of Layer Normalization for training stability.

## Content

### [01_transformer_decoder_architecture.ipynb](./01_transformer_decoder_architecture.ipynb)
A deep dive into the model components with Mermaid diagrams and stability experiments.
- **Multi-Head Attention**: Q, K, V projections and the attention mechanism.
- **Transformer Block**: The residual stream and the role of Pre-norm.
- **Output Stage**: Final LayerNorm (`ln_f`) and the Linear Header (`lm_head`).

## Shared Resources

This lesson uses the shared `nanoGPT-lab` module.

```bash
# To install dependencies and set up the environment
make install
```

Then start Jupyter Lab:

```bash
make lab
```
