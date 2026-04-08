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


MODELS_DIR = "pkl_data"
# Maps trait label in filename -> output column name (OCEAN order)


def get_one_embedding_for_text(text, ):
    hidden_dim = 768

    dataset = ReviewDataset(
        text = text,
        tokenizer=tokenizer,
        token_length=2000
    )
    # num_workers=0 is essential on Windows — worker processes each load
    # a copy of the model into RAM and never release it



    with torch.no_grad():
        output = bert_model(
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # hidden_states: tuple of (n_layers+1) tensors shaped (batch, seq_len, hidden_dim)
        # Index 0 = embedding layer, 1..12 = transformer layers
        # CLS token is position 0 in the sequence
        cls_embedding = output.hidden_states[embed_layer][:, 0, :].cpu().numpy()
        return cls_embedding
TRAIT_MAP = {
    "OPN": "O_openness",
    "CON": "C_conscientiousness",
    "EXT": "E_extraversion",
    "AGR": "A_agreeableness",
    "NEU": "N_neuroticism",
}
def load_keras_models():
    """
    Loads the 5 .h5 Keras models from models_dir.
    Returns a dict: {trait_label -> keras model}
    """
    import tensorflow as tf

    models = {}
    for trait in TRAIT_MAP:
        path = os.path.join(MODELS_DIR, f"MLP_LM_{trait}.h5")
        if not os.path.exists(path):
            print(f"[ERROR] Missing model file: {path}")
            print("Make sure you ran MLP_LM.py with -save_model yes and the")
            print(f"5 .h5 files are in: {MODELS_DIR}")
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


if __name__ == "__main__":
    user_ids, embeddings = extract_embeddings(
        df, tokenizer, bert_model,
        args.token_length, args.batch_size, args.embed_layer, device,
    )

    models = load_keras_models()