import re
from collections import Counter

class SimpleTextTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>"}
        
    def build_vocab(self, sentences, max_vocab_size=10000):
        words = []
        for sentence in sentences:
            clean_text = re.sub(r"[^\w\s]", "", sentence.lower())
            words.extend(clean_text.split())
            
        most_common = Counter(words).most_common(max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.vocab[word] = idx
            self.inverse_vocab[idx] = word
            
    def encode(self, sentence, max_len=32):
        clean_text = re.sub(r"[^\w\s]", "", sentence.lower())
        tokens = clean_text.split()
        
        # Convert words to IDs, use 1 (<UNK>) if word isn't in vocab
        encoded = [self.vocab.get(word, 1) for word in tokens[:max_len]]
        
        # Padding
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
            
        return encoded

    def __len__(self):
        return len(self.vocab)