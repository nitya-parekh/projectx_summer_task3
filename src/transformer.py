import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""Attention(Q,K,V) = softmax((QK^T) / sqrt(d_k))V]"""

class ScaledDotProductAttention(nn.Module):             #raw mathematic formula; no memory
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        # q, k, v are the Query, Key, and Value tensors
        d_k = q.size(-1) # this is the size of the dimension we are dividing by
                

        # 1. Multiply Q and the transpose of K (the QK^T part of the formula)
        # We use matrix multiplication (matmul)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # 2. Scale the scores down by the square root of d_k
        scores = scores / math.sqrt(d_k)
        
        # 3. Apply the mask (Optional, but required later for the Decoder)
        # This hides future words so the model can't "cheat" and look ahead
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 4. Turn the scores into percentages that add up to 100%
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. Multiply the percentages by the Values (V)
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    #giving model actual "brain cells" (weights) that it will learn and adjust during training.
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        # d_model is the total size of your word/atom meaning (e.g., 64 numbers)
        # num_heads is how many researchers you want (e.g., 8)
        self.d_model = d_model
        self.num_heads = num_heads
        
        # We divide the total size by the number of heads (64 / 8 = 8)
        # Each researcher only gets a slice of the data to work with.
        self.d_k = d_model // num_heads # // because we need a whole number of dimensions for each head, not decimals.
        
        # These linear layers are like "glasses" for each researcher.
        # They reshape the raw input into custom Queries, Keys, and Values.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # This is the math engine we built in the previous step!
        self.attention = ScaledDotProductAttention()
        
        # The final output layer to blend all the researchers' notes back together
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # Grab the batch size (how many SMILES strings we are processing at once)
        batch_size = q.size(0)
        
        # 1. Pass the inputs through the Linear layers
        # 2. .view() chops the data into 8 separate pieces (our 8 heads)
        # 3. .transpose() moves the "heads" dimension so it sits next to the batch size
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 4. Feed all 8 chopped-up pieces into our Attention Math Engine at once
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # 5. The researchers are done. We transpose the data back to its original shape
        # .contiguous() forces the computer's memory to line up cleanly
        # .view() mashes the 8 separate pieces back into one single block (d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 6. Pass the mashed-up data through the final linear layer
        return self.fc_out(output)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 1. Create an empty matrix of zeros to hold our positional timestamps
        pe = torch.zeros(max_len, d_model)
        
        # 2. Create a column of numbers from 0 to max_len (representing the atom's position)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Calculate the different speeds for our "clock hands"
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. Apply the fast-spinning Sine to the even columns (0, 2, 4...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 5. Apply the slower Cosine to the odd columns (1, 3, 5...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 6. Add a "Batch" dimension to the front so it aligns with our input data
        pe = pe.unsqueeze(0)
        
        # 7. Save this matrix in PyTorch's memory as a fixed constant
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Simply add the positional timestamps to our atom data!
        # We slice pe[:, :x.size(1)] so we only grab the exact amount of timestamps we need.
        x = x + self.pe[:, :x.size(1)]
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # First linear layer expands the dimension (e.g., 64 to 2048)
        self.w_1 = nn.Linear(d_model, d_ff)
        # Second linear layer shrinks it back down (e.g., 2048 to 64)
        self.w_2 = nn.Linear(d_ff, d_model)
        # Dropout randomly zeroes out some data to prevent the model from memorizing
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pass through w_1, apply ReLU activation, apply dropout, then pass through w_2
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # The multi-head attention mechanism we built earlier
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # The feed-forward network defined just above
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization stabilizes the network by keeping numbers balanced
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers for regularization during training
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # --- Part 1: Self-Attention and Add & Norm ---
        # 1. Run the data through the attention mechanism
        attn_output = self.self_attn(x, x, x, mask)
        # 2. Add the original input (x) to the output (residual connection), then normalize
        x = self.layer_norm1(x + self.dropout1(attn_output))
        
        # --- Part 2: Feed-Forward and Add & Norm ---
        # 3. Run the normalized data through the feed-forward network
        ff_output = self.feed_forward(x)
        # 4. Add the previous state (x) to the new output, then normalize again
        x = self.layer_norm2(x + self.dropout2(ff_output))
        
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Self-attention for the target sequence
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Cross-attention to look back at the Encoder's findings
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        # Standard feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Three normalization layers for stability
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 1. Masked Self-Attention (prevents looking at future tokens)
        attn_output1 = self.self_attn(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout1(attn_output1))
        
        # 2. Cross-Attention (Query from Decoder, Keys/Values from Encoder)
        attn_output2 = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.layer_norm2(x + self.dropout2(attn_output2))
        
        # 3. Feed-Forward Network
        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + self.dropout3(ff_output))
        
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8, num_layers=6, d_ff=1024, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Embedding layers to turn atom IDs into d_model sized vectors
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional Encoding mechanism
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Create multiple stacked Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # Final linear layer to predict the next token
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 1. Embed and add positional encoding to source and target strings
        src_emb = self.dropout(self.positional_encoding(self.src_embedding(src)))
        tgt_emb = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))
        
        # 2. Pass source through all Encoder layers
        enc_output = src_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        # 3. Pass target and encoder output through all Decoder layers
        dec_output = tgt_emb
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        # 4. Generate final token predictions
        return self.fc_out(dec_output)