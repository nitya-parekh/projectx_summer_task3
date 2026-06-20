# Transformer: SMILES String Parser

This repository contains a scratch-built PyTorch implementation of the Transformer architecture (*Attention Is All You Need*). It is designed to process and translate chemical SMILES strings into numerical tokens, demonstrating sequence-to-sequence learning.

## Project Structure
* `src/` - Contains the raw PyTorch neural network, tokenizer, and training loop.
* `results/` - Contains terminal output logs proving the loss metrics during training.
* `PAPER_NOTES.md` - Analysis of the original research paper.

## Prerequisites
Ensure you have Python 3.12+ installed.
```bash
pip install torch numpy