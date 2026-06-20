import torch
import torch.nn as nn
import torch.optim as optim
from tokenizer import SimpleTextTokenizer
from transformer import TransformerClassifier

# 1. Prepare Dummy Text Dataset (Sentiment Analysis)
data = [
    ("This movie was absolutely amazing and beautiful", 1),
    ("I loved the acting and the script was brilliant", 1),
    ("An incredible masterpiece of cinema", 1),
    ("That film was a complete waste of time", 0),
    ("Horrible acting and terrible directing", 0),
    ("Boring plot and very poorly shot movie", 0)
]
sentences, labels = zip(*data)

# 2. Setup Pipeline Components
tokenizer = SimpleTextTokenizer()
tokenizer.build_vocab(sentences)

X = torch.tensor([tokenizer.encode(s) for s in sentences])
Y = torch.tensor(labels)

model = TransformerClassifier(vocab_size=len(tokenizer), num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 3. Training Loop
print("--- Starting Sentence Classifier Training Execution ---")
for epoch in range(1, 41):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/40] | Optimization Loss Value: {loss.item():.4f}")

print("\n[+] Training loop executed successfully. Logs generated.")