# Root Makefile to manage all lessons

.PHONY: install format clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies and setup kernels for all lessons"
	@echo "  make format   - Format code in all lessons using Ruff"
	@echo "  make clean    - Remove all virtual environments and kernels"

install:
	@echo "Installing Lesson 1..."
	$(MAKE) -C "Lesson 1. Micrograd" install
	@echo "Installing Lesson 2..."
	$(MAKE) -C "Lesson 2. nanoGPT" install

format:
	@echo "Formatting Lesson 1..."
	$(MAKE) -C "Lesson 1. Micrograd" format
	@echo "Formatting Lesson 2..."
	$(MAKE) -C "Lesson 2. nanoGPT" format

clean:
	@echo "Cleaning Lesson 1..."
	$(MAKE) -C "Lesson 1. Micrograd" clean
	@echo "Cleaning Lesson 2..."
	$(MAKE) -C "Lesson 2. nanoGPT" clean
