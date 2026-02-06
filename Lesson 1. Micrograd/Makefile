.PHONY: install clean

PYTHON=3.13
KERNEL_NAME=micrograd
DISPLAY_NAME="Python 3.13 (Micrograd)"

install:
	uv sync
	uv run python -m ipykernel install --user --name=$(KERNEL_NAME) --display-name $(DISPLAY_NAME)
run:
	uv run jupyter notebook

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf .venv
	jupyter kernelspec uninstall $(KERNEL_NAME) -y
