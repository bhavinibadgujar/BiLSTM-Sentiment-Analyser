# ============================================================
# train.py — BiLSTM Sentiment Model Training Script
#
# Usage:
#   python train.py
#
# What it does:
#   1. Loads dataset.csv
#   2. Tokenises text with NLTK (via utils.py)
#   3. Builds a vocabulary
#   4. Encodes sequences and creates DataLoaders
#   5. Trains a BiLSTM classifier (defined in model.py)
#   6. Saves the best checkpoint to bilstm_model.pth
# ============================================================

import os
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model import BiLSTMSentimentClassifier
from utils import (
    IDX2LABEL,
    LABEL2IDX,
    SentimentDataset,
    Vocabulary,
    build_vocab_from_csv,
    tokenize,
)

# ── Hyper-parameters ─────────────────────────────────────────
CSV_PATH    = "dataset.csv"
CHECKPOINT  = "bilstm_model.pth"

EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.4
MAX_LEN     = 64

BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 1e-3
MIN_FREQ    = 1          # minimum token frequency for vocab inclusion
TEST_SIZE   = 0.15
RANDOM_SEED = 42

# ── Device setup ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ── Helper: compute accuracy ──────────────────────────────────
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ── Main training routine ─────────────────────────────────────
def main():
    # ── 1. Load data ─────────────────────────────────────────
    print(f"[INFO] Loading dataset from '{CSV_PATH}' …")
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].str.strip().str.lower()
    print(f"[INFO] {len(df)} samples | label distribution:\n{df['label'].value_counts()}\n")

    # ── 2. Build vocabulary ───────────────────────────────────
    print("[INFO] Building vocabulary …")
    vocab = build_vocab_from_csv(CSV_PATH, min_freq=MIN_FREQ)
    print(f"[INFO] Vocabulary size: {len(vocab)}\n")

    # ── 3. Train / validation split ───────────────────────────
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    print(f"[INFO] Train samples: {len(train_df)} | Val samples: {len(val_df)}\n")

    # ── 4. Datasets & DataLoaders ─────────────────────────────
    train_dataset = SentimentDataset(train_df, vocab, max_len=MAX_LEN)
    val_dataset   = SentimentDataset(val_df,   vocab, max_len=MAX_LEN)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ── 5. Model, loss, optimiser ─────────────────────────────
    model = BiLSTMSentimentClassifier(
        vocab_size  = len(vocab),
        embed_dim   = EMBED_DIM,
        hidden_dim  = HIDDEN_DIM,
        output_dim  = len(LABEL2IDX),
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        pad_idx     = 0,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)

    print("[INFO] Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {total_params:,}\n")

    # ── 6. Training loop ──────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train phase ───────────────────────────────────────
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for token_ids, labels in train_loader:
            token_ids = token_ids.to(DEVICE)
            labels    = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(token_ids)             # forward pass
            loss   = criterion(logits, labels)    # compute loss
            loss.backward()                       # back-propagate
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()

            train_loss += loss.item()
            train_acc  += accuracy(logits, labels)

        train_loss /= len(train_loader)
        train_acc  /= len(train_loader)

        # ── Validation phase ──────────────────────────────────
        model.eval()
        val_loss, val_acc = 0.0, 0.0

        with torch.no_grad():
            for token_ids, labels in val_loader:
                token_ids = token_ids.to(DEVICE)
                labels    = labels.to(DEVICE)
                logits    = model(token_ids)
                loss      = criterion(logits, labels)
                val_loss += loss.item()
                val_acc  += accuracy(logits, labels)

        val_loss /= len(val_loader)
        val_acc  /= len(val_loader)

        # ── Learning-rate scheduling ───────────────────────────
        scheduler.step(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # ── Save best checkpoint ──────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "vocab_size":  len(vocab),
                    "embed_dim":   EMBED_DIM,
                    "hidden_dim":  HIDDEN_DIM,
                    "output_dim":  len(LABEL2IDX),
                    "num_layers":  NUM_LAYERS,
                    "dropout":     DROPOUT,
                    "max_len":     MAX_LEN,
                    "val_acc":     best_val_acc,
                },
                CHECKPOINT,
            )
            print(f"  ✓ Saved new best checkpoint (val_acc={best_val_acc:.4f})")

    print(f"\n[DONE] Training complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"[DONE] Model checkpoint saved to '{CHECKPOINT}'")


if __name__ == "__main__":
    main()
