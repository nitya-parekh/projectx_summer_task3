# Paper Notes: "Attention Is All You Need" (Vaswani et al., 2017)

## Central Claim
The paper claims that a model built entirely on self-attention (with no recurrence or convolution)
achieves better translation quality than the best RNN/CNN-based seq2seq models of the time,
while being significantly faster to train due to full parallelism.

Reported result: 28.4 BLEU on WMT 2014 English→German, beating the previous best (ConvS2S at 26.36)
by over 2 BLEU points, at a fraction of the training cost.

## Core Architecture
The Transformer consists of:
- Encoder: stack of N=6 identical layers, each with (1) multi-head self-attention and (2) a
  position-wise feed-forward network, both wrapped in residual connections + LayerNorm
- Decoder: stack of N=6 layers with an extra cross-attention sublayer attending to encoder output
- Input: learned token embeddings + sinusoidal positional encodings (since attention has no
  built-in notion of order)
- Multi-Head Attention: h=8 parallel attention heads, d_model=512, d_k=d_v=64 per head
- FFN: inner dimension d_ff=2048, ReLU activation
- Dropout: p=0.1 applied after attention weights and each sublayer (Section 5.4)

## Dataset, Metric, and Baseline
- Dataset: WMT 2014 English→German (4.5M sentence pairs), tokenized with byte-pair encoding
- Metric: BLEU score (measures n-gram overlap between generated and reference translations)
- Baseline: ConvS2S (26.36 BLEU), Deep-Att + PosUnk (26.30 BLEU)
- The Transformer (big) achieves 28.4 BLEU, a new state of the art at the time

## My Adaptation and Deviations
Since translation requires a decoder and sequence generation, which would need weeks of training
on WMT 2014 to verify, I adapted the encoder to a classification task (SST-2 sentiment analysis)
to verify that the core architecture — multi-head self-attention, positional encoding, residual
connections, and the FFN sublayer — works correctly.

Deliberate simplifications due to compute:
- Encoder only (no decoder) — classification doesn't require sequence generation
- N=2 layers instead of N=6 (paper default), d_model=128 instead of 512
- Word-level tokenizer instead of BPE (subword tokenization)
- Metric changes from BLEU to accuracy on SST-2 validation set
- Expected result: ~80-85% accuracy on SST-2 val (BERT-scale models hit ~93%, but a small
  2-layer encoder trained for 10 epochs is reasonably expected in this range)