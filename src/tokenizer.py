import torch

class SMILESTokenizer:
    def __init__(self):
        # The mandatory special tokens every Transformer needs
        # PAD: Fills empty space so all sentences are the same length
        # SOS: Start of Sequence (tells the Decoder to start generating)
        # EOS: End of Sequence (tells the Decoder to stop generating)
        # UNK: Unknown character fallback
        self.special_tokens = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        
        # We will build these dictionaries when we load our dataset
        self.char_to_id = self.special_tokens.copy()
        self.id_to_char = {v: k for k, v in self.special_tokens.items()}
        self.vocab_size = len(self.char_to_id)

    def build_vocab(self, smiles_list):
        # Scans a list of SMILES strings and assigns a unique ID to every new character
        for smiles in smiles_list:
            for char in smiles:
                if char not in self.char_to_id:
                    self.char_to_id[char] = self.vocab_size
                    self.id_to_char[self.vocab_size] = char
                    self.vocab_size += 1

    def encode(self, smiles_string, max_length=None):
        # 1. Start with the <SOS> token
        encoded = [self.char_to_id['<SOS>']]
        
        # 2. Convert each chemical character into its numerical ID
        for char in smiles_string:
            encoded.append(self.char_to_id.get(char, self.char_to_id['<UNK>']))
            
        # 3. End with the <EOS> token
        encoded.append(self.char_to_id['<EOS>'])
        
        # 4. If a max_length is provided, chop it or pad it with <PAD> (0)
        if max_length is not None:
            if len(encoded) > max_length:
                encoded = encoded[:max_length] # Chop if too long
            else:
                encoded = encoded + [self.char_to_id['<PAD>']] * (max_length - len(encoded)) # Pad if too short
                
        # Return as a PyTorch tensor ready for the neural network
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, token_ids):
        # Converts the network's numerical output back into a readable SMILES string
        decoded = []
        for token in token_ids:
            # Convert tensor to standard Python integer
            token_val = token.item() if isinstance(token, torch.Tensor) else token
            
            # Stop translating if we hit the End of Sequence token or Padding
            if token_val == self.char_to_id['<EOS>'] or token_val == self.char_to_id['<PAD>']:
                break
                
            # Skip the Start of Sequence token in the final readable text
            if token_val != self.char_to_id['<SOS>']:
                decoded.append(self.id_to_char.get(token_val, '?'))
                
        # Join the characters back into a single string
        return "".join(decoded)