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


MODELS_DIR = '../../'
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