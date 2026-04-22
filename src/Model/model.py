import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np

class sentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=60):
        self.texts = [self._pad_or_trunc([word2idx.get(w, word2idx['<UNK>']) for w in text.split()], max_len) for text in texts]
        self.labels = labels
        if vocab is None:
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        return {word: idx for idx, (word, _) in enumerate(counter.most_common())}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = [self.vocab.get(word, 0) for word in text.split()]
        return torch.tensor(indices), torch.tensor(label)