import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from tokenizer import SimpleTextTokenizer
from transformer import TransformerClassifier

# 1. Load real SST-2 dataset
print("Loading SST-2 dataset...")
dataset = load_dataset("stanfordnlp/sst2")
train_data = dataset["train"]
val_data   = dataset["validation"]

train_data = train_data.select(range(8000))
val_data   = val_data.select(range(500))

train_sentences = train_data["sentence"]
train_labels    = train_data["label"]
val_sentences   = val_data["sentence"]
val_labels      = val_data["label"]
# 2. Build tokenizer on training sentences only
tokenizer = SimpleTextTokenizer()
tokenizer.build_vocab(train_sentences, max_vocab_size=10000)

MAX_LEN = 64

X_train = torch.tensor([tokenizer.encode(s, MAX_LEN) for s in train_sentences])
Y_train = torch.tensor(train_labels)
X_val   = torch.tensor([tokenizer.encode(s, MAX_LEN) for s in val_sentences])
Y_val   = torch.tensor(val_labels)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)

# 3. Model
model = TransformerClassifier(
    vocab_size=len(tokenizer),
    num_classes=2,
    d_model=128,
    num_heads=4,
    d_ff=256,
    max_len=MAX_LEN,
    num_layers=2,
    dropout=0.1
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. Training loop
print("--- Starting Training ---")
for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation accuracy after each epoch
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).argmax(dim=1)
        val_acc = (val_preds == Y_val).float().mean().item()

    print(f"Epoch [{epoch}/20] | Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.4f}")

print("\n[+] Done. Save a screenshot of these logs for your results/ folder.")