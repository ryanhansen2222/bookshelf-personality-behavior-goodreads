"""
predict_ocean.py
----------------
Runs BERT embedding extraction + MLP inference on your Goodreads
user_aggregated_reviews.csv and outputs per-user OCEAN scores.

Does both steps in one pass:
  1. Loads user text blobs from your CSV
  2. Passes them through BERT (bert-base-uncased) to get CLS embeddings
  3. Runs the 5 Essays-trained Keras MLPs to produce O/C/E/A/N scores
  4. Writes user_ocean_scores.csv

The .h5 model files come from running:
  python finetune_models/MLP_LM.py -save_model yes
in the Mehta repo. They live at:
  pkl_data/finetune_mlp_lm/MLP_LM_EXT.h5  etc.

Usage:
  # Full run (BERT extraction + scoring):
  python predict_ocean.py \
      --input      aggregated/user_aggregated_reviews.csv \
      --models_dir personality-prediction/pkl_data/finetune_mlp_lm/ \
      --out_dir    aggregated/

  # Skip re-extraction if you already have embeddings.pkl:
  python predict_ocean.py \
      --input          aggregated/user_aggregated_reviews.csv \
      --models_dir     personality-prediction/pkl_data/finetune_mlp_lm/ \
      --embeddings_pkl aggregated/user_embeddings.pkl \
      --out_dir        aggregated/

  # Extract embeddings only (no scoring yet):
  python predict_ocean.py \
      --input        aggregated/user_aggregated_reviews.csv \
      --extract_only \
      --out_dir      aggregated/

Requirements (all available in the conda environment):
  pip install torch transformers tensorflow pandas numpy tqdm
"""

import argparse
import gc
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",          required=True,
                   help="Path to user_aggregated_reviews.csv")
    p.add_argument("--models_dir",     default=None,
                   help="Directory containing the 5 MLP_LM_*.h5 files")
    p.add_argument("--out_dir",        default=".",
                   help="Output directory (default: current directory)")
    p.add_argument("--batch_size",     type=int, default=4,
                   help="BERT batch size. Keep at 4 on CPU (default: 4)")
    p.add_argument("--token_length",   type=int, default=512)
    p.add_argument("--embed_layer",    type=int, default=12,
                   help="Which BERT layer to use (1-12 for bert-base). Default=12 (last).")
    p.add_argument("--extract_only",   action="store_true",
                   help="Only extract BERT embeddings; skip MLP scoring.")
    p.add_argument("--embeddings_pkl", default=None,
                   help="Path to existing user_embeddings.pkl to skip BERT re-extraction.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    """Tokenises user text blobs for the PyTorch DataLoader."""

    def __init__(self, texts, user_ids, tokenizer, token_length):
        self.texts        = texts
        self.user_ids     = user_ids
        self.tokenizer    = tokenizer
        self.token_length = token_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            self.user_ids[idx],
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0),
        )


# ---------------------------------------------------------------------------
# BERT extraction
# ---------------------------------------------------------------------------

def extract_embeddings(df, tokenizer, bert_model, token_length, batch_size, embed_layer, device):
    """
    Passes all user text blobs through BERT and returns CLS embeddings.

    Streams embeddings to a memmap temp file on disk so RAM stays flat —
    the old approach of appending numpy arrays each batch caused a steady
    memory leak because Python's GC doesn't release them promptly.

    Returns:
        user_ids:   list[str]
        embeddings: np.ndarray of shape (N, 768)
    """
    n_users    = len(df)
    hidden_dim = 768

    # Allocate on disk up front — only the current batch is ever in RAM
    tmp      = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
    tmp_path = tmp.name
    tmp.close()
    emb_mmap = np.memmap(tmp_path, dtype="float32", mode="w+", shape=(n_users, hidden_dim))

    dataset = ReviewDataset(
        texts=df["text"].tolist(),
        user_ids=df["user_id"].tolist(),
        tokenizer=tokenizer,
        token_length=token_length,
    )
    # num_workers=0 is essential on Windows — worker processes each load
    # a copy of the model into RAM and never release it
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    bert_model.eval()
    all_user_ids = []
    row_idx      = 0

    print(f"\nExtracting BERT embeddings (layer {embed_layer}) for {n_users:,} users...")
    print("Streaming to disk — RAM usage will stay flat.")

    for user_ids_batch, input_ids, attention_mask in tqdm(loader, unit="batch"):
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            output = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # hidden_states: tuple of (n_layers+1) tensors shaped (batch, seq_len, hidden_dim)
        # Index 0 = embedding layer, 1..12 = transformer layers
        # CLS token is position 0 in the sequence
        cls_embedding = output.hidden_states[embed_layer][:, 0, :].cpu().numpy()

        n = cls_embedding.shape[0]
        emb_mmap[row_idx : row_idx + n] = cls_embedding
        emb_mmap.flush()  # write through to disk immediately

        all_user_ids.extend(list(user_ids_batch))
        row_idx += n

        # Explicitly drop every reference so Python GC can reclaim the memory
        del input_ids, attention_mask, output, cls_embedding
        gc.collect()

    # Read back into a normal array, then delete the temp file
    embeddings = np.array(emb_mmap)
    del emb_mmap
    os.remove(tmp_path)

    print(f"Embedding matrix shape: {embeddings.shape}")
    return all_user_ids, embeddings


# ---------------------------------------------------------------------------
# Keras MLP scoring
# ---------------------------------------------------------------------------

# Maps trait label in filename -> output column name (OCEAN order)
TRAIT_MAP = {
    "OPN": "O_openness",
    "CON": "C_conscientiousness",
    "EXT": "E_extraversion",
    "AGR": "A_agreeableness",
    "NEU": "N_neuroticism",
}

def load_keras_models(models_dir):
    """
    Loads the 5 .h5 Keras models from models_dir.
    Returns a dict: {trait_label -> keras model}
    """
    import tensorflow as tf

    models = {}
    for trait in TRAIT_MAP:
        path = os.path.join(models_dir, f"MLP_LM_{trait}.h5")
        if not os.path.exists(path):
            print(f"[ERROR] Missing model file: {path}")
            print("Make sure you ran MLP_LM.py with -save_model yes and the")
            print(f"5 .h5 files are in: {models_dir}")
            sys.exit(1)
        models[trait] = tf.keras.models.load_model(path)
        print(f"  Loaded {trait}: {path}")

    return models


def score_with_keras(models, embeddings):
    """
    Runs inference with the 5 Keras models.

    Each model outputs shape (N, 2) — logits over [low, high].
    We apply softmax and take column 1 (probability of trait being high).

    Returns a dict: {trait_label -> np.ndarray of shape (N,)}
    """
    import tensorflow as tf

    scores = {}
    for trait, model in models.items():
        logits = model.predict(embeddings, verbose=0)  # (N, 2)
        probs  = tf.nn.softmax(logits, axis=1).numpy()
        scores[trait] = probs[:, 1]

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load CSV ---
    print(f"\nLoading {args.input} ...")
    df = pd.read_csv(args.input)

    missing = {"user_id", "text"} - set(df.columns)
    if missing:
        print(f"[ERROR] CSV is missing columns: {missing}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    print(f"Users loaded: {len(df):,}")

    # --- Load or extract BERT embeddings ---
    if args.embeddings_pkl:
        print(f"\nLoading embeddings from {args.embeddings_pkl} ...")
        with open(args.embeddings_pkl, "rb") as f:
            saved = pickle.load(f)
        user_ids   = saved["user_ids"]
        embeddings = saved["embeddings"]
        print(f"Loaded embeddings for {len(user_ids):,} users, shape={embeddings.shape}")

    else:
        print("\nLoading bert-base-uncased (downloads ~440MB on first run)...")
        tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        bert_model = bert_model.to(device)

        user_ids, embeddings = extract_embeddings(
            df, tokenizer, bert_model,
            args.token_length, args.batch_size, args.embed_layer, device,
        )

        # Always save embeddings — BERT extraction is the expensive step
        emb_path = os.path.join(args.out_dir, "user_embeddings.pkl")
        with open(emb_path, "wb") as f:
            pickle.dump({"user_ids": user_ids, "embeddings": embeddings}, f)
        print(f"\nEmbeddings saved to: {emb_path}")
        print("Tip: re-run with --embeddings_pkl to skip BERT extraction next time.")

        if args.extract_only:
            print("\n--extract_only set. Done. Re-run without it to produce OCEAN scores.")
            return

    # --- MLP scoring ---
    if not args.models_dir:
        print("\n[INFO] No --models_dir given. Cannot produce OCEAN scores.")
        print("Pass --models_dir pointing to the folder with MLP_LM_EXT.h5 etc.")
        return

    print(f"\nLoading Keras MLP models from: {args.models_dir}")
    models = load_keras_models(args.models_dir)

    print("\nRunning MLP inference...")
    scores = score_with_keras(models, embeddings)

    # --- Build output DataFrame ---
    results = pd.DataFrame({"user_id": user_ids})
    for trait, col in TRAIT_MAP.items():
        results[col] = scores[trait].round(4)

    # Merge back with review_count / word_count from input CSV
    meta_cols = [c for c in ["user_id", "review_count", "word_count"] if c in df.columns]
    results = results.merge(df[meta_cols], on="user_id", how="left")

    # Save
    out_path = os.path.join(args.out_dir, "user_ocean_scores.csv")
    results.to_csv(out_path, index=False)

    trait_cols = list(TRAIT_MAP.values())
    print(f"\nOCEAN scores written to: {out_path}")
    print(f"Users scored: {len(results):,}")
    print("\nTrait score summary (0-1 scale, higher = stronger trait signal):")
    print(results[trait_cols].describe().round(3).to_string())
    print("\nSample rows:")
    print(results.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
