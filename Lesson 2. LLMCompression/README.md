# Lesson 2: LLM Compression

This lesson is divided into two main stages covering the research foundations and large-scale application of Large Language Model compression techniques.

## Structure

### [Stage 1 - LLM Compression Research (nanoGPT)](./Stage%201%20-%20LLM%20Compression%20Research%20(nanoGPT))
In this stage, we primarily explore LLM compression using the **nanoGPT** architecture. We focus on the fundamental mathematical principles behind various techniques.

- **Steps 1-4:** Analysis, Quantization, and initial steps.
- **Steps 5-16:** LoRA Fine-tuning, SVD Decomposition, Pruning, AWQ Scaling, N:M Sparsity, Distillation, Mixed Precision, SmoothQuant, and KV Cache optimization.

### [Stage 2 - Large Scale & SOTA (Llama 3)](./Stage%202%20-%20Large%20Scale%20&%20SOTA%20(Llama%203))
In the second part, we transition to **Llama 3 (1B parameters)** to explore modern, industrial-grade compression methods and SOTA (State Of The Art) advancements.

- **Quantization:** QAT, PTQ (Weight-only, Weight-Activation), KV Cache quantization.
- **Pruning:** Unstructured, Structured, and Semi-structured techniques.
- **Distillation:** Black-box and White-box approaches.
- **Advanced Topics:** Low-rank factorization, Mixture of Experts (MoE), and 2026/SOTA advancements.

## Setup & Shared Resources
The following shared resources are located at the root of the directory:
- `src/`: Core nanoGPT model implementation.
- `train.py`, `generate.py`, `chat.py`, `analyze.py`: Scripts for nanoGPT training and inference.
- `input.txt`: Training data (Shakespeare).
- `model_*.pt`: Pre-trained model checkpoints.

> [!NOTE]
> All notebooks are configured to look for the `src` module in the parent directory.
