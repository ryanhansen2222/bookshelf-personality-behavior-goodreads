"""
Takes 10,000 reviews from goodreads_reviews_dedup.json.gz, runs each through a 
modified version of unseen_predictor.predict(), and saves OCEAN scores to a CSV.

Usage:
    python run_goodreads_pipeline.py

Output:
    personality_predictions.csv   (user_id, review_id, EXT, NEU, AGR, CON, OPN)

Notes:
    Assumes that yashsmehta's personality-prediction repo has been cloned, the model
    has already been trained and saved, this script is in the root of the
    personality-prediction repo, and goodreads_reviews_dedup.json.gz is located
    at REVIEWS_FILE path below.
"""

import os
import gzip
import json
import random
import csv
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress tensorflow log output
sys.path.insert(0, os.getcwd())            # imports should come from personality-prediction root

# ── Setup ────────────────────────────────────────────────────────────────────────
REVIEWS_FILE   = "data/goodreads_reviews_dedup.json.gz"
OUTPUT_CSV     = "personality_predictions.csv"

SAMPLE_SIZE    = 10_000
RANDOM_SEED    = 42
MIN_TEXT_LEN   = 50                # skip reviews shorter than 50 chars

EMBED          = "bert-base"
OP_DIR         = "pkl_data/"       # must contain finetune_mlp_lm/
FINETUNE_MODEL = "MLP_LM"
DATASET        = "essays"
TOKEN_LENGTH   = 512
OCEAN_TRAITS = ["EXT", "NEU", "AGR", "CON", "OPN"]


# ── Load BERT and trait models once, reuse across all reviews ────────────────────
_tokenizer       = None
_bert_model      = None
_finetune_models = None

def _load_models():
    global _tokenizer, _bert_model, _finetune_models

    if _tokenizer is not None:
        return   # already loaded

    print("Loading BERT model")
    from unseen_predictor import get_bert_model, load_finetune_model, DEVICE

    _tokenizer, _bert_model = get_bert_model(EMBED)
    _bert_model.to(DEVICE)
    _bert_model.eval()

    print("Loading personality trait models")
    _finetune_models = load_finetune_model(OP_DIR, FINETUNE_MODEL, DATASET)
    print("All models loaded.\n")


# ── Prediction ───────────────────────────────────────────────────────────────────

def predict_one(text: str) -> dict[str, float]:
    """
    Runs a single review text through the full pipeline

    Returns key-value pairs for traits, e.g.
        {"EXT": 0.61, "NEU": 0.38, "AGR": 0.54, "CON": 0.47, "OPN": 0.72}

    Note:
        unseen_predictor prints results but returns None and loads BERT on every run.
        This is an altered version so we can save the values and only load BERT once.
    """
    import numpy as np
    from unseen_predictor import extract_bert_features, softmax
    import utils.dataset_processors as dataset_processors

    # Preprocess text
    text_pre = dataset_processors.preprocess_text(text)

    # Get BERT embedding
    embeddings = extract_bert_features(
        text_pre, _tokenizer, _bert_model, TOKEN_LENGTH
    )

    # Run each trait model and collect score
    predictions = {}
    for trait, trait_model in _finetune_models.items():
        prediction = trait_model.predict(embeddings)
        prediction = softmax(prediction)
        prediction = prediction[0][1]
        predictions[trait] = float(prediction)

    return predictions


# ── Reservoir sampler ─────────────────────────────────────────────────────────────

def reservoir_sample(filepath: str, k: int, min_len: int, seed: int) -> list[dict]:
    """
    Randomly samples reviews from the json.gz file.
    Uses reservoir sampling to guarantee equal probability for each review.
    """
    rng    = random.Random(seed)
    bucket = []
    n_seen = 0

    print(f"Scanning {filepath} (this may take a few minutes)...")
    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip trivial reviews
            text = record.get("review_text", "").strip()
            if len(text) < min_len:
                continue

            # Skip known non-English reviews
            lang = record.get("language_code", "")
            if lang and lang not in ("", "en", "eng", "en-US", "en-GB"):
                continue

            n_seen += 1

            if len(bucket) < k:
                bucket.append(record)
            else:
                # Replace a random earlier entry with prob k/n_seen
                idx = rng.randint(0, n_seen - 1)
                if idx < k:
                    bucket[idx] = record

            # Update on progress every 500_000 reviews
            if n_seen % 500_000 == 0:
                print(f"  scanned {n_seen:,} valid reviews | reservoir: {len(bucket)}")

    print(f"Scan complete. {n_seen:,} qualifying reviews found, {len(bucket)} sampled.\n")
    return bucket


# ── Main pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline():
    # Check that reviews file exists
    if not Path(REVIEWS_FILE).exists():
        sys.exit(
            f"ERROR: Reviews dataset not found at '{REVIEWS_FILE}'.\n"
        )

    # Check that saved model directory exists
    expected_model_dir = Path(OP_DIR) / f"finetune_{FINETUNE_MODEL.lower()}"
    if not expected_model_dir.is_dir():
        sys.exit(
            f"ERROR: Trained model directory not found: {expected_model_dir}\n"
            "You must train the model first.\n"
        )

    # ── Step 1: Sample reviews ────────────────────────────────────────────────────
    print("=" * 60)
    print(f"STEP 1/3  Sampling {SAMPLE_SIZE:,} reviews")
    print("=" * 60)
    sample = reservoir_sample(REVIEWS_FILE, SAMPLE_SIZE, MIN_TEXT_LEN, RANDOM_SEED)

    # ── Step 2: Load models (once) then predict ───────────────────────────────────
    print("=" * 60)
    print("STEP 2/3  Running personality predictions")
    print("=" * 60)
    _load_models()

    results  = []
    n_errors = 0
    t_start  = time.time()

    for i, record in enumerate(sample):
        user_id   = record.get("user_id", "")
        review_id = record.get("review_id", "")
        text      = record.get("review_text", "").strip()

        try:
            scores = predict_one(text)
        except Exception as e:
            print(f"  [WARN] review {review_id}: {e}", file=sys.stderr)
            n_errors += 1
            continue

        row = {"user_id": user_id, "review_id": review_id}
        row.update(scores)      # adds EXT, NEU, AGR, CON, OPN
        results.append(row)

        # Progress + ETA
        if (i + 1) % 100 == 0:
            elapsed  = time.time() - t_start
            per_item = elapsed / (i + 1)
            remaining = per_item * (len(sample) - i - 1)
            print(
                f"  {i + 1}/{len(sample)} done | "
                f"{elapsed:.0f}s elapsed | "
                f"~{remaining/60:.1f} min remaining"
            )

    print(f"\nPredictions complete: {len(results)} succeeded, {n_errors} skipped.\n")

    # ── Step 3: Save to CSV ───────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3/3  Save results")
    print("=" * 60)

    fieldnames = ["user_id", "review_id"] + OCEAN_TRAITS
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results):,} rows → {OUTPUT_CSV}")
    print("Done!")


if __name__ == "__main__":
    run_pipeline()
