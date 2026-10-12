import torch
import torch.nn.functional as F
from typing import List, Optional


class AminoAcidTokenizer:
    def __init__(self):
        amino_acids = [
            "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
        ]
        self.special_symbols = {"<PAD>": 0, "<UNK>": 21, "<BOS>": 22, "<EOS>": 23}
        self.vocab = {amino_acid: index for index, amino_acid in enumerate(amino_acids, start=1)}
        self.vocab.update(self.special_symbols)

    def encode(self, sequence: str, max_len: Optional[int] = None) -> torch.Tensor:
        tokens = [self.vocab.get(amino_acid, self.vocab["<UNK>"]) for amino_acid in sequence.upper()]
        encoded = torch.tensor(tokens, dtype=torch.long)
        if max_len is not None:
            if encoded.numel() < max_len:
                encoded = F.pad(encoded, (0, max_len - encoded.numel()), value=self.vocab["<PAD>"])
            else:
                encoded = encoded[:max_len]
        return encoded

    def batch_encode(
        self, sequences: List[str], max_len: Optional[int] = None, padding: bool = True
    ) -> torch.Tensor:
        encoded = [self.encode(sequence, max_len if padding else None) for sequence in sequences]
        if padding and max_len is not None:
            return torch.stack(encoded)
        return torch.nn.utils.rnn.pad_sequence(
            encoded, batch_first=True, padding_value=self.vocab["<PAD>"]
        )
