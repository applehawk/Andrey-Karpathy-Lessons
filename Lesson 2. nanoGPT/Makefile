.PHONY: install format clean

KERNEL_NAME=nanogpt
DISPLAY_NAME="Python 3.13 (nanoGPT)"

install:
	uv sync
	uv run python -m ipykernel install --user --name=$(KERNEL_NAME) --display-name $(DISPLAY_NAME)

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf .venv uv.lock
	jupyter kernelspec uninstall $(KERNEL_NAME) -y
