.PHONY: install format clean solve-paths lab

lab:
	uv run jupyter lab

KERNEL_NAME=llmcompression
DISPLAY_NAME="Python 3.13 (LLMCompression)"

install:
	uv sync
	uv pip install -e .
	uv run python -m ipykernel install --user --name=$(KERNEL_NAME) --display-name $(DISPLAY_NAME)

train:
	uv run python train.py

generate:
	uv run python generate.py

chat:
	uv run python chat.py

format:
	uv run ruff format .
	uv run ruff check --fix .

# Fix paths in notebooks after moving to subdirectories
solve-paths:
	uv run python fix_paths.py

clean:
	rm -rf .venv uv.lock
	-jupyter kernelspec uninstall $(KERNEL_NAME) -y
