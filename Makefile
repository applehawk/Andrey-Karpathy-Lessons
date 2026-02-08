# Root Makefile to manage all lessons и nanoGPT resources

.PHONY: install format clean help lab train generate chat

# Default target
help:
	@echo "Available commands:"
	@echo "  make install  - Install environment (uv sync) and setup jupyter kernel"
	@echo "  make lab      - Run Jupyter Lab"
	@echo "  make format   - Format code using Ruff"
	@echo "  make train    - Train nanoGPT model"
	@echo "  make generate - Generate text from trained nanoGPT"
	@echo "  make chat     - Interactive chat with nanoGPT"
	@echo "  make clean    - Remove virtual environment and cached files"

install:
	@echo "Installing project dependencies..."
	uv sync
	uv pip install -e .
	@echo "Setting up Jupyter kernel..."
	uv run python -m ipykernel install --user --name=llm-colab --display-name "Python 3.13 (LLM-Colab)"

lab:
	uv run jupyter lab

format:
	uv run ruff format .
	uv run ruff check --fix .

train:
	uv run python nanoGPT-lab/train.py

generate:
	uv run python nanoGPT-lab/generate.py

chat:
	uv run python nanoGPT-lab/chat.py

clean:
	rm -rf .venv uv.lock
	-jupyter kernelspec uninstall llm-colab -y
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
