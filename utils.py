# ============================================================
# utils.py — Text Preprocessing & Dataset Utilities
# Handles tokenisation (NLTK), vocabulary building, text
# encoding, and the PyTorch Dataset class used by train.py
# and app.py.
# ============================================================

import re
import string
from collections import Counter
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import torch
from torch.utils.data import Dataset

# Download required NLTK resources the first time this module is imported.
nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("stopwords",    quiet=True)
nltk.download("wordnet",      quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Constants ────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"   # index 0 — used to pad sequences to equal length
UNK_TOKEN = "<UNK>"   # index 1 — replaces tokens not seen during training

# Mapping from raw label strings to integer class indices
LABEL2IDX: Dict[str, int] = {
    "positive": 0,
    "neutral":  1,
    "negative": 2,
}
IDX2LABEL: Dict[int, str] = {v: k for k, v in LABEL2IDX.items()}

# ── Text cleaner ─────────────────────────────────────────────
_STOP_WORDS   = set(stopwords.words("english"))
_LEMMATIZER   = WordNetLemmatizer()
_PUNCT_TABLE  = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """
    Lower-case, strip punctuation & digits, remove stop-words,
    then lemmatise each remaining token.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)          # remove digits
    text = text.translate(_PUNCT_TABLE)      # remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # normalise whitespace
    return text


def tokenize(text: str) -> List[str]:
    """
    Clean text then tokenise with NLTK word_tokenize.
    Stop-words are removed and each token is lemmatised.
    """
    text   = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and tok.strip()
    ]
    return tokens


# ── Vocabulary ───────────────────────────────────────────────
class Vocabulary:
    """
    Builds and stores a word-to-index mapping.

    Usage:
        vocab = Vocabulary()
        vocab.build(list_of_token_lists, min_freq=1)
        idx   = vocab["hello"]         # → int
        token = vocab.idx2token[idx]   # → str
    """

    def __init__(self):
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self._counter:  Counter        = Counter()

    def build(self, tokenised_texts: List[List[str]], min_freq: int = 1) -> None:
        """
        Count all tokens, then create the vocabulary by keeping
        tokens that appear at least *min_freq* times.
        """
        for tokens in tokenised_texts:
            self._counter.update(tokens)

        # Always place special tokens at fixed indices
        special = [PAD_TOKEN, UNK_TOKEN]
        vocab_tokens = special + [
            tok for tok, freq in self._counter.most_common()
            if freq >= min_freq
        ]

        self.token2idx = {tok: idx for idx, tok in enumerate(vocab_tokens)}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}

    def __len__(self) -> int:
        return len(self.token2idx)

    def __getitem__(self, token: str) -> int:
        """Return index for token, falling back to <UNK> if unseen."""
        return self.token2idx.get(token, self.token2idx[UNK_TOKEN])

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a token list to a list of integer indices."""
        return [self[tok] for tok in tokens]


# ── Sequence padding helper ───────────────────────────────────
def pad_sequence(
    encoded: List[int],
    max_len: int,
    pad_idx: int = 0,
) -> List[int]:
    """
    Truncate or right-pad *encoded* so its length equals *max_len*.
    """
    if len(encoded) >= max_len:
        return encoded[:max_len]
    return encoded + [pad_idx] * (max_len - len(encoded))


# ── PyTorch Dataset ───────────────────────────────────────────
class SentimentDataset(Dataset):
    """
    Wraps a pandas DataFrame (columns: 'text', 'label') into a
    PyTorch Dataset that returns (token_ids_tensor, label_tensor) pairs.

    Args:
        df      : DataFrame with 'text' and 'label' columns
        vocab   : pre-built Vocabulary instance
        max_len : sequence length after padding/truncation
    """

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int = 64):
        self.vocab   = vocab
        self.max_len = max_len
        self.samples: List[Tuple[List[int], int]] = []

        for _, row in df.iterrows():
            tokens  = tokenize(str(row["text"]))
            encoded = vocab.encode(tokens)
            padded  = pad_sequence(encoded, max_len)
            label   = LABEL2IDX.get(str(row["label"]).strip().lower(), 1)
            self.samples.append((padded, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids, label = self.samples[idx]
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(label,     dtype=torch.long),
        )


# ── Convenience: build vocab directly from CSV ───────────────
def build_vocab_from_csv(csv_path: str, min_freq: int = 1) -> Vocabulary:
    """
    Read *csv_path*, tokenise every text row, and return a fitted Vocabulary.
    Used by both train.py and app.py to ensure identical token → index mapping.
    """
    df = pd.read_csv(csv_path)
    tokenised = [tokenize(str(t)) for t in df["text"]]
    vocab = Vocabulary()
    vocab.build(tokenised, min_freq=min_freq)
    return vocab
