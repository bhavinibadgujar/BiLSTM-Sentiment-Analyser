# ============================================================
# model.py — BiLSTM Sentiment Classifier
# Defines the neural network architecture used for training
# and inference throughout the project.
# ============================================================

import torch
import torch.nn as nn


class BiLSTMSentimentClassifier(nn.Module):
    """
    Bidirectional LSTM model for multi-class sentiment classification.

    Architecture:
        Embedding → Dropout → BiLSTM → Dropout → FC → Output
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 3,      # positive / neutral / negative
        num_layers: int = 2,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ):
        super(BiLSTMSentimentClassifier, self).__init__()

        # ── Embedding layer ──────────────────────────────────
        # Maps integer token IDs → dense vectors.
        # padding_idx=0 keeps the <PAD> embedding zeroed out.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # ── Dropout (applied after embedding) ────────────────
        self.embed_dropout = nn.Dropout(dropout)

        # ── Bidirectional LSTM ────────────────────────────────
        # batch_first=True  →  input/output shape: (B, T, H)
        # bidirectional=True  →  hidden dim is doubled internally
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Dropout (applied after LSTM) ─────────────────────
        self.lstm_dropout = nn.Dropout(dropout)

        # ── Fully-connected classification head ──────────────
        # hidden_dim * 2  because BiLSTM concatenates both directions
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    # ── Forward pass ─────────────────────────────────────────
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: LongTensor of shape (batch, seq_len)
        Returns:
            logits: FloatTensor of shape (batch, output_dim)
        """
        # 1. Embed tokens  →  (B, T, embed_dim)
        embedded = self.embed_dropout(self.embedding(token_ids))

        # 2. Run BiLSTM
        #    lstm_out  →  (B, T, hidden_dim * 2)
        #    hidden    →  (num_layers * 2, B, hidden_dim)  [not used directly]
        lstm_out, (hidden, _) = self.lstm(embedded)

        # 3. Pool: concatenate the final forward and backward hidden states
        #    hidden[-2]  →  last layer forward direction  (B, hidden_dim)
        #    hidden[-1]  →  last layer backward direction (B, hidden_dim)
        pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, hidden_dim * 2)

        # 4. Apply dropout then project to class logits
        logits = self.fc(self.lstm_dropout(pooled))           # (B, output_dim)

        return logits
