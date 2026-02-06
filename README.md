# Andrey Karpathy's Neural Networks Lessons

This repository contains my progress and experiments through Andrey Karpathy's deep learning series.

## 📚 Project Structure

- **[Lesson 1: Micrograd](./Lesson%201.%20Micrograd)** - Building a tiny autograd engine from scratch.
- **[Lesson 2: nanoGPT](./Lesson%202.%20nanoGPT)** - Implementing a GPT-style Transformer and exploring **LLM Compression** (Quantization, LoRA, Pruning).

## 🛠 Tech Stack

- **Python 3.13**
- **[uv](https://github.com/astral-sh/uv)** - Ultra-fast Python package manager.
- **PyTorch** - For matrix-based operations and LLM experiments.
- **Ruff** - Fast linting and formatting for both Python files and Jupyter Notebooks.
- **direnv** - Automatic virtual environment activation upon entering directory.

## 🚀 Getting Started

Each lesson is self-contained with its own dependencies managed via `uv`.

1. **Install uv** (if you haven't):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Setup a lesson**:
   ```bash
   cd "Lesson 2. nanoGPT"
   make install
   ```

3. **Format code**:
   ```bash
   make format
   ```

## 📈 Current Focus: LLM Compression

I am currently working on Lesson 2, transitioning from basic Transformer implementation to advanced compression techniques:
- [x] Basic GPT Architecture
- [ ] Manual INT8 Quantization
- [ ] LoRA (Low-Rank Adaptation)
- [ ] Pruning & Saliency Analysis
