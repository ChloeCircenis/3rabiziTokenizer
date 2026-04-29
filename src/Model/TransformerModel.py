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

def make_batches(tokens_list, labels_list, batch_size, pad_id, shuffle=True):
    data = list(zip(tokens_list, labels_list))

    if shuffle:
        random.shuffle(data)

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_tokens, batch_labels = zip(*batch)

        max_len = max(len(x) for x in batch_tokens)

        padded = []
        for tokens in batch_tokens:
            padded.append(tokens + [pad_id] * (max_len - len(tokens)))

        input_ids = torch.tensor(padded, dtype=torch.long)
        labels = torch.tensor(batch_labels, dtype=torch.long)

        yield input_ids, labels

class TransformerSentimentModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_classes=3,
        max_len=256,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        pad_idx=0,
        dropout=0.1
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.max_len = max_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.position_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_len:
            input_ids = input_ids[:, :self.max_len]
            seq_len = self.max_len

        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        padding_mask = input_ids == self.pad_idx

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

        logits = self.classifier(pooled)
        return logits


def encode_texts(texts, tokenizer):
    return [tokenizer.encode(t).ids for t in texts]


def train_one_epoch(model, data_tokens, data_labels, loss_func, optimizer, pad_id, batch_size):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in make_batches(data_tokens, data_labels, batch_size, pad_id, shuffle=True):
        optimizer.zero_grad()

        logits = model(input_ids)
        loss = loss_func(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, data_tokens, data_labels, loss_func, pad_id, batch_size):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for input_ids, labels in make_batches(data_tokens, data_labels, batch_size, pad_id, shuffle=False):
        logits = model(input_ids)
        loss = loss_func(logits, labels)

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # store for F1
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    avg_loss = total_loss / total
    accuracy = correct / total

    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, f1


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
pad_id = tokenizer.token_to_id("<pad>")
if pad_id is None:
    pad_id = 0

model = TransformerSentimentModel(vocab_size=vocab_size,pad_idx=pad_id)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
batch_size = 32
num_epochs = 5

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model,
        train_tokens,
        train_sent,
        loss_func,
        optimizer,
        pad_id,
        batch_size
    )

    test_loss, test_acc, test_f1 = evaluate(
        model,
        test_tokens,
        test_sent,
        loss_func,
        pad_id,
        batch_size
    )

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        f"Test F1: {test_f1:.4f}"
    )