# Root Makefile to manage all lessons

.PHONY: install format clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies and setup kernels for all lessons"
	@echo "  make format   - Format code in all lessons using Ruff"
	@echo "  make train-gpt - Train the nanoGPT model"
	@echo "  make generate-gpt - Generate text from trained nanoGPT"
	@echo "  make chat-gpt - Interactive chat with nanoGPT"
	@echo "  make clean    - Remove all virtual environments and kernels"

install: submodules
	@echo "Installing Lesson 1..."
	$(MAKE) -C "Lesson 1. Micrograd" install
	@echo "Installing Lesson 2..."
	$(MAKE) -C "Lesson 2. LLMCompression" install

submodules:
	@echo "Initializing submodules..."
	git submodule update --init --recursive

format:
	@echo "Formatting Lesson 1..."
	$(MAKE) -C "Lesson 1. Micrograd" format
	@echo "Formatting Lesson 2..."
	$(MAKE) -C "Lesson 2. LLMCompression" format

clean:
	@echo "Cleaning Lesson 1..."
	$(MAKE) -C "Lesson 1. Micrograd" clean
	@echo "Cleaning Lesson 2..."
	$(MAKE) -C "Lesson 2. LLMCompression" clean

train-gpt:
	$(MAKE) -C "Lesson 2. LLMCompression" train

generate-gpt:
	$(MAKE) -C "Lesson 2. LLMCompression" generate

chat-gpt:
	$(MAKE) -C "Lesson 2. LLMCompression" chat
