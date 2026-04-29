import csv
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
import sentencepiece as spm


label_map = {-1: 0, 0: 1, 1: 2}
inv_label_map = {0: -1, 1: 0, 2: 1}


def read_csv_file(file_path):
    text_list = []
    sent_list = []

    with open(file_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text_list.append(row["text"])
            if "label" in row:
                raw_label = int(row["label"])
                sent_list.append(label_map[raw_label])

    if len(sent_list) == 0:
        return text_list
    return text_list, sent_list


def normalize_text(text_list):
    normalized = []
    for text in text_list:
        text = text.lower()
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
        normalized.append(text)
    return normalized


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.classifier = nn.Linear(embed_dim, 3)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)

        if attention_mask is None:
            pooled = emb.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            emb = emb * mask
            pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

        logits = self.classifier(pooled)   # [B, 3]
        return logits


def encode_texts(texts, tokenizer):
    return [tokenizer.encode(t).ids for t in texts]


def train_one_epoch(model, data_tokens, data_labels, loss_func, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    combined = list(zip(data_tokens, data_labels))
    random.shuffle(combined)

    for tokens, label in combined:
        if len(tokens) == 0:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([label], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)   # [1, 3]
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_label = logits.argmax(dim=-1).item()
        correct += int(pred_label == label)
        total += 1

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, data_tokens, data_labels, loss_func):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for tokens, label in zip(data_tokens, data_labels):
        if len(tokens) == 0:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([label], dtype=torch.long)

        logits = model(input_ids, attention_mask)
        loss = loss_func(logits, labels)

        total_loss += loss.item()
        pred_label = logits.argmax(dim=-1).item()
        correct += int(pred_label == label)
        total += 1
        all_preds.append(pred_label)
        all_labels.append(label)

    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total, correct / total, f1


# -----------------------------
# Paths
# -----------------------------
train_file = "/Users/jakeburton/Desktop/NLP/NLP grad version/3rabiziTokenizer/src/SAMAWEL JABALLI dataset/Train (1).csv"
BPE_vocab_file_4 = "/Users/jakeburton/Desktop/NLP/NLP grad version/3rabiziTokenizer/src/BPE/BPE_4.json"
BPE_vocab_file_8 = "/Users/jakeburton/Desktop/NLP/NLP grad version/3rabiziTokenizer/src/BPE/BPE_8.json"
BPE_vocab_file_16 = "/Users/jakeburton/Desktop/NLP/NLP grad version/3rabiziTokenizer/src/BPE/BPE_16.json"
BPE_vocab_file_28 = "/Users/jakeburton/Desktop/NLP/NLP grad version/3rabiziTokenizer/src/BPE/BPE_28.json"
BPE_vocab_file_44 = "/Users/jakeburton/Desktop/NLP/NLP grad version/3rabiziTokenizer/src/BPE/BPE_44.json"
Unigram_vocab_file = "/Users/jakeburton/Desktop/NLP/NLP grad version/3rabiziTokenizer/src/UnigramModel/spmthirty.model"
tokenizer_path = BPE_vocab_file_4 # Change this to the desired tokenizer path (BPE or Unigram)
# -----------------------------
# Load data
# -----------------------------
text, sent = read_csv_file(train_file)

train_text, test_text, train_sent, test_sent = train_test_split(
    text, sent, test_size=0.2, random_state=42, stratify=sent
)

train_text = normalize_text(train_text)
test_text = normalize_text(test_text)

print("Train label counts:", Counter(train_sent))
print("Test label counts:", Counter(test_sent))

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = Tokenizer.from_file(tokenizer_path) # for BPE tokenizer
#tokenizer = spm.SentencePieceProcessor()        # for Unigram tokenizer
#tokenizer.load(tokenizer_path)                  # ------------

train_tokens = encode_texts(train_text, tokenizer)
test_tokens = encode_texts(test_text, tokenizer)

vocab_size = len(tokenizer.get_vocab())
embed_dim = 128
pad_id = tokenizer.token_to_id("<pad>")
if pad_id is None:
    pad_id = 0

model = SentimentModel(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_id)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_tokens, train_sent, loss_func, optimizer
    )
    test_loss, test_acc, test_f1 = evaluate(
        model, test_tokens, test_sent, loss_func
    )

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | " 
        f"Test F1: {test_f1:.4f}"
    )