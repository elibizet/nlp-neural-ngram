"""
Bengio-style feedforward neural n-gram language model
using Pytorch and embeddings that are learned vector 
representations of words, where semantically similar 
words have similar vectors.”

Run it with:

python3 neural_ngram.py \
  --train_path /home/dsv/robe/lis060/data/english-train.txt.gz \
  --test_path /home/dsv/robe/lis060/data/english-test-1k.txt.gz \
  --save_embeddings embeddings.vec \
  --save_model neural_ngram.pt
  
  How to check the vectors: python3 wvv.py embeddings.vec Friday
  
  The model does the following:

    Input: 4 previous words
    Each word gets a 100-dim embedding
    Concatenate the 4 embeddings
    Feed through a hidden layer of size 80 with tanh
    Predict next word over the whole vocabulary
    Ignore training cases where target is <UNK>
    
  """

import argparse
import gzip
import math
import random
from collections import Counter
from pathlib import Path
from tokenizer import tokenize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PAD = "<PAD>"
EOS = "<EOS>"
UNK = "<UNK>"


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def read_sentences(path: str):
    """
    Reads one sentence per line, tokenized by whitespace.
    Appends <EOS> to each sentence.
    """
    sentences = []
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = tokenize(line)
            tokens.append(EOS)
            sentences.append(tokens)
    return sentences


def build_vocab(sentences, min_freq=5):
    counter = Counter()
    for sent in sentences:
        counter.update(sent)

    vocab = [PAD, EOS, UNK]
    for word, freq in counter.items():
        if word not in {PAD, EOS, UNK} and freq >= min_freq:
            vocab.append(word)

    stoi = {word: i for i, word in enumerate(vocab)}
    itos = {i: word for word, i in stoi.items()}
    return stoi, itos, counter


def numericalize_sentences(sentences, stoi):
    unk_id = stoi[UNK]
    return [[stoi.get(tok, unk_id) for tok in sent] for sent in sentences]


class NGramDataset(Dataset):
    """
    Context of 4 previous words -> next word target.
    Excludes examples where the target is <UNK>.
    Allows <UNK> in the context.
    """

    def __init__(self, sentence_ids, pad_id, unk_id, context_size=4):
        self.examples = []

        for sent in sentence_ids:
            padded = [pad_id] * context_size + sent
            for i in range(context_size, len(padded)):
                context = padded[i - context_size:i]
                target = padded[i]
                if target == unk_id:
                    continue
                self.examples.append((context, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context, target = self.examples[idx]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


class NeuralNGramLM(nn.Module):
    """
    Bengio-style feedforward neural n-gram language model:
    embeddings -> concat -> linear -> tanh -> linear -> vocab logits
    """

    def __init__(self, vocab_size, emb_dim=100, hidden_dim=80, context_size=4, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(context_size * emb_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, context_size)
        emb = self.embedding(x)                # (batch, context, emb_dim)
        emb = emb.reshape(emb.size(0), -1)     # (batch, context * emb_dim)
        h = self.activation(self.fc1(emb))
        logits = self.fc2(h)                   # (batch, vocab_size)
        return logits


def build_dataset(path, stoi=None, min_freq=5):
    sentences = read_sentences(path)

    if stoi is None:
        stoi, itos, counter = build_vocab(sentences, min_freq=min_freq)
    else:
        itos = {i: w for w, i in stoi.items()}

    sentence_ids = numericalize_sentences(sentences, stoi)

    dataset = NGramDataset(
        sentence_ids=sentence_ids,
        pad_id=stoi[PAD],
        unk_id=stoi[UNK],
        context_size=4,
    )
    return dataset, stoi, itos


def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            logits = model(contexts)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            total_tokens += targets.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def save_embeddings(path, embedding_matrix, itos):
    """
    Saves in the Section 2.3-style format:
    first line: vocab_size dim
    remaining lines: word val1 val2 ... valD
    """
    vocab_size, dim = embedding_matrix.size()

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{vocab_size} {dim}\n")
        for i in range(vocab_size):
            word = itos[i]
            vec = " ".join(f"{x:.6f}" for x in embedding_matrix[i].tolist())
            f.write(f"{word} {vec}\n")


def train(args):
    set_seed(args.seed)

    print("Reading training data...")
    train_dataset, stoi, itos = build_dataset(
        args.train_path,
        stoi=None,
        min_freq=args.min_freq,
    )

    print(f"Vocabulary size: {len(stoi)}")
    print(f"Training examples: {len(train_dataset)}")

    print("Reading test data...")
    test_dataset, _, _ = build_dataset(
        args.test_path,
        stoi=stoi,
        min_freq=args.min_freq,
    )
    print(f"Test examples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    model = NeuralNGramLM(
        vocab_size=len(stoi),
        emb_dim=100,
        hidden_dim=80,
        context_size=4,
        pad_idx=stoi[PAD],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # sanity check before training
    test_loss, test_ppl = evaluate(model, test_loader, device)
    print(f"Before training | test NLL/token = {test_loss:.4f} | ppl = {test_ppl:.2f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for step, (contexts, targets) in enumerate(train_loader, start=1):
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(contexts)
            loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += targets.size(0)

            if step % 100 == 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(avg_loss)
                print(
                    f"Epoch {epoch} | step {step} | "
                    f"train NLL/token = {avg_loss:.4f} | ppl = {ppl:.2f}"
                )

        train_avg_loss = total_loss / total_tokens
        train_ppl = math.exp(train_avg_loss)

        test_loss, test_ppl = evaluate(model, test_loader, device)

        print(
            f"End epoch {epoch} | "
            f"train NLL/token = {train_avg_loss:.4f} | train ppl = {train_ppl:.2f} | "
            f"test NLL/token = {test_loss:.4f} | test ppl = {test_ppl:.2f}"
        )

    if args.save_model:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
            },
            args.save_model,
        )
        print(f"Saved model to {args.save_model}")

    save_embeddings(
        args.save_embeddings,
        model.embedding.weight.detach().cpu(),
        itos,
    )
    print(f"Saved embeddings to {args.save_embeddings}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--save_embeddings", type=str, default="embeddings.vec")
    parser.add_argument("--save_model", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()