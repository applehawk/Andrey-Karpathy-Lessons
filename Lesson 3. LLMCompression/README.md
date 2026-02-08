# Lesson 3: LLM Compression

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

## Installation

To set up the environment and install dependencies:

```bash
make install
```

This will:
1. Install all Python dependencies using `uv`
2. Install the project in editable mode (so `src` module is importable)
3. Register the Jupyter kernel

Then start Jupyter Lab:

```bash
make lab
```

## Setup & Shared Resources

All shared nanoGPT resources are located in the `nanoGPT-lab/` directory:

### Core Modules (`nanoGPT-lab/src/`)
- **`model.py`** - Pure GPT architecture (Karpathy style)
  - Classes: `GPTLanguageModel`, `Block`, `MultiHeadAttention`, `Head`, `FeedFoward`
  - Hyperparameters defined at module level
- **`utils.py`** - Data loading and training utilities
  - Tokenizer: `encode()`, `decode()`
  - Data: `get_batch()`, `estimate_loss()`
- **`inference.py`** - Extended generation functions
  - `generate_stream()`, `generate_text()`, `generate_text_stream()`

### Scripts & Data
- `train.py`, `generate.py`, `chat.py`, `analyze.py` - Training and inference scripts
- `input.txt` - Training data (Shakespeare)
- `model_ckpt.pt` - Pre-trained checkpoint

Each Stage directory contains a symlink `nanoGPT-lab` pointing to the shared resources.

> [!NOTE]
> The project uses a clean modular structure:
> - **`model.py`**: Architecture only (like Karpathy's original)
> - **`utils.py`**: Data and training helpers
> - **`inference.py`**: Extended inference features
> 
> Install in editable mode with `make install` to import from anywhere: `from src.model import GPTLanguageModel`

