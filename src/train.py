import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from tokenizer import SMILESTokenizer

# --- 1. Toy Dataset ---
# A small dataset to prove the model works on your laptop without crashing
smiles_data = [
    "CC(=O)Oc1ccccc1C(=O)O",         # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
    "CCO",                           # Ethanol
    "C1=CC=CC=C1"                    # Benzene
]

# --- 2. Initialize Tokenizer ---
tokenizer = SMILESTokenizer()
tokenizer.build_vocab(smiles_data)
vocab_size = tokenizer.vocab_size

# Encode the data and pad to a maximum length of 30 characters
max_len = 30
encoded_data = torch.stack([tokenizer.encode(s, max_len) for s in smiles_data])

# For auto-regressive training, inputs are everything except the last token
# Targets are everything except the first token (<SOS>)
src = encoded_data[:, :-1]
tgt = encoded_data[:, :-1]
targets = encoded_data[:, 1:] # What the model should actually predict

# --- 3. Masking Functions ---
def make_src_mask(src, pad_idx):
    # Hides padding tokens from the encoder so it doesn't process blank space
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def make_tgt_mask(tgt, pad_idx):
    # Hides padding AND future tokens from the decoder (the blindfold)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len))).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return tgt_mask

# --- 4. Initialize the Model ---
# Using small dimensions (d_model=64, layers=2) to ensure it runs fast locally
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=64,
    num_heads=8,
    num_layers=2, 
    d_ff=256,
    max_len=max_len,
    dropout=0.1
)

# --- 5. Training Setup ---
pad_idx = tokenizer.char_to_id['<PAD>']
# CrossEntropyLoss automatically ignores padding tokens when calculating the score
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
# Adam optimizer updates the model's weights to minimize the loss
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# --- 6. The Training Loop ---
epochs = 50
print("Starting training...")

for epoch in range(epochs):
    model.train()
    # Wipe the gradients clean from the previous step
    optimizer.zero_grad()
    
    # Generate the safety masks
    src_mask = make_src_mask(src, pad_idx)
    tgt_mask = make_tgt_mask(tgt, pad_idx)
    
    # Forward pass: model makes its predictions
    output = model(src, tgt, src_mask, tgt_mask)
    
    # Reshape output and targets to a flat list to calculate loss
    output_flat = output.contiguous().view(-1, vocab_size)
    targets_flat = targets.contiguous().view(-1)
    
    # Calculate how wrong the predictions were
    loss = criterion(output_flat, targets_flat)
    
    # Backward pass: model learns from its mistakes
    loss.backward()
    optimizer.step()
    
    # Print the logs every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

print("Training complete! Save these logs for your results folder.")