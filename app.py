# ============================================================
# app.py — Streamlit Sentiment Analysis Web App
#
# Run locally:
#   streamlit run app.py
#
# Run on Render (production):
#   streamlit run app.py --server.port 10000 --server.address 0.0.0.0
#
# The app loads the trained BiLSTM checkpoint and lets the
# user type any text to receive a live sentiment prediction.
# ============================================================

import os

import torch
import streamlit as st

from model import BiLSTMSentimentClassifier
from utils import (
    IDX2LABEL,
    build_vocab_from_csv,
    pad_sequence,
    tokenize,
)

# ── Configuration ────────────────────────────────────────────
CSV_PATH   = "dataset.csv"
CHECKPOINT = "bilstm_model.pth"
MAX_LEN    = 64
DEVICE     = torch.device("cpu")   # Streamlit Cloud / Render → CPU only

# ── Emoji & colour map for labels ────────────────────────────
LABEL_META = {
    "positive": {"emoji": "😊", "colour": "#2ecc71", "desc": "Positive"},
    "neutral":  {"emoji": "😐", "colour": "#f39c12", "desc": "Neutral"},
    "negative": {"emoji": "😞", "colour": "#e74c3c", "desc": "Negative"},
}

# ── Streamlit page config ─────────────────────────────────────
st.set_page_config(
    page_title="BiLSTM Sentiment Analyser",
    page_icon="🧠",
    layout="centered",
)


# ── Cached resource loaders (run once per session) ───────────
@st.cache_resource(show_spinner="Loading vocabulary …")
def load_vocab():
    """Rebuild the vocabulary from the training CSV."""
    return build_vocab_from_csv(CSV_PATH, min_freq=1)


@st.cache_resource(show_spinner="Loading model …")
def load_model(vocab_size: int):
    """
    Load the BiLSTM model from the saved checkpoint.
    Falls back gracefully if the checkpoint is missing.
    """
    if not os.path.exists(CHECKPOINT):
        return None, None

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    model = BiLSTMSentimentClassifier(
        vocab_size  = ckpt.get("vocab_size",  vocab_size),
        embed_dim   = ckpt.get("embed_dim",   128),
        hidden_dim  = ckpt.get("hidden_dim",  256),
        output_dim  = ckpt.get("output_dim",  3),
        num_layers  = ckpt.get("num_layers",  2),
        dropout     = ckpt.get("dropout",     0.4),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()   # switch to inference mode (disables dropout)

    return model, ckpt


# ── Inference helper ──────────────────────────────────────────
def predict(text: str, model: BiLSTMSentimentClassifier, vocab) -> dict:
    """
    Tokenise → encode → pad → run model → return prediction dict.

    Returns:
        {
          "label":       "positive" | "neutral" | "negative",
          "confidence":  float  (0-100 %),
          "probabilities": {"positive": float, "neutral": float, "negative": float}
        }
    """
    tokens   = tokenize(text)
    encoded  = vocab.encode(tokens)
    padded   = pad_sequence(encoded, MAX_LEN)

    tensor   = torch.tensor([padded], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)                       # (1, 3)
        probs  = torch.softmax(logits, dim=1)[0]     # (3,)

    # IDX2LABEL: {0: "positive", 1: "neutral", 2: "negative"}
    pred_idx = probs.argmax().item()
    label    = IDX2LABEL[pred_idx]
    conf     = probs[pred_idx].item() * 100

    prob_map = {IDX2LABEL[i]: round(probs[i].item() * 100, 1) for i in range(3)}
    return {"label": label, "confidence": conf, "probabilities": prob_map}


# ══════════════════════════════════════════════════════════════
# ── Streamlit UI ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────
st.title("🧠 BiLSTM Sentiment Analyser")
st.markdown(
    "A **Bidirectional LSTM** neural network trained to classify text as "
    "**Positive**, **Neutral**, or **Negative**."
)
st.divider()

# ── Load resources ────────────────────────────────────────────
vocab          = load_vocab()
model, ckpt    = load_model(len(vocab))

# ── Sidebar: model info ───────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    if ckpt:
        st.metric("Vocab Size",      f"{ckpt.get('vocab_size', len(vocab)):,}")
        st.metric("Embedding Dim",   ckpt.get("embed_dim",  128))
        st.metric("Hidden Dim",      ckpt.get("hidden_dim", 256))
        st.metric("BiLSTM Layers",   ckpt.get("num_layers", 2))
        st.metric("Max Seq Length",  ckpt.get("max_len",    64))
        best_acc = ckpt.get("val_acc", None)
        if best_acc:
            st.metric("Best Val Accuracy", f"{best_acc*100:.1f}%")
    else:
        st.warning(
            "No checkpoint found.\n\n"
            "Please train the model first:\n```\npython train.py\n```"
        )
    st.divider()
    st.markdown(
        "**Classes**\n"
        "- 😊 Positive\n"
        "- 😐 Neutral\n"
        "- 😞 Negative"
    )

# ── Missing checkpoint guard ──────────────────────────────────
if model is None:
    st.error(
        "**Model checkpoint not found.**\n\n"
        "Run `python train.py` in the project directory to train the model, "
        "then restart the app."
    )
    st.stop()

# ── Text input area ───────────────────────────────────────────
st.subheader("📝 Enter Your Text")
user_text = st.text_area(
    label       = "Type or paste any text below:",
    placeholder = "e.g. I absolutely love this product!",
    height      = 140,
    label_visibility="collapsed",
)

# ── Example buttons ───────────────────────────────────────────
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😊 Positive example"):
        user_text = "This product is absolutely fantastic! I am thrilled with the quality."

with col2:
    if st.button("😐 Neutral example"):
        user_text = "The item arrived on time and works as described. Nothing special."

with col3:
    if st.button("😞 Negative example"):
        user_text = "Terrible quality. It broke after one day and support ignored me."

# ── Run inference ─────────────────────────────────────────────
if st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text before analysing.")
    else:
        with st.spinner("Analysing …"):
            result = predict(user_text, model, vocab)

        label  = result["label"]
        conf   = result["confidence"]
        meta   = LABEL_META[label]

        st.divider()
        st.subheader("📊 Prediction")

        # ── Main result badge ─────────────────────────────────
        st.markdown(
            f"""
            <div style="
                background: {meta['colour']}22;
                border: 2px solid {meta['colour']};
                border-radius: 12px;
                padding: 20px 30px;
                text-align: center;
                margin-bottom: 20px;
            ">
                <span style="font-size: 3rem;">{meta['emoji']}</span><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: {meta['colour']};">
                    {meta['desc']}
                </span><br>
                <span style="font-size: 1rem; color: #888;">
                    Confidence: {conf:.1f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Probability breakdown ─────────────────────────────
        st.markdown("**Probability Breakdown**")
        for lbl in ["positive", "neutral", "negative"]:
            p  = result["probabilities"][lbl]
            m  = LABEL_META[lbl]
            st.markdown(
                f"{m['emoji']} **{m['desc']}** — {p:.1f}%"
            )
            st.progress(p / 100)

        # ── Tokenisation peek ─────────────────────────────────
        with st.expander("🔬 Show tokenised input"):
            tokens = tokenize(user_text)
            st.write(tokens if tokens else ["(no tokens after preprocessing)"])

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with PyTorch + NLTK + Streamlit · "
    "BiLSTM model trained on a 3-class sentiment dataset."
)
