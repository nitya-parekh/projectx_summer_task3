# Transformer Sentence Classifier (Task 3)

This is my submission for Task 3. I implemented the core architecture from the famous "Attention Is All You Need" paper (Vaswani et al., 2017) completely from scratch using PyTorch. 

Instead of building a massive translator, I adapted the encoder part of the paper to build a simple text sentence classifier (sentiment analysis). 

## How the Code is Structured
I didn't use HuggingFace or any pre-made transformer libraries because I wanted to learn how the math actually works under the hood.
* `src/tokenizer.py`: My custom text tokenizer. It cleans up sentences, splits them into lowercase words, builds a vocabulary, and pads everything so the arrays match.
* `src/transformer.py`: The actual neural network. It sets up the Positional Encodings (the sine/cosine waves), Multi-Head Self-Attention, and the Encoder layers.
* `src/train.py`: A basic training loop with some sample sentences to prove the model is actually minimizing loss and learning.

## How to Setup & Run This on Your Machine

```bash
pip install torch datasets
cd src
python train.py
```

Expected output: loss decreasing each epoch, validation accuracy printed after each epoch.
Save a screenshot of the terminal output and put it in results/.