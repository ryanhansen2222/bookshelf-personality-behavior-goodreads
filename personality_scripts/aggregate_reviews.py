"""
aggregate_reviews.py
--------------------
Aggregates Goodreads Fantasy & Paranormal review data into per-user text blobs
compatible with the Mehta et al. LM_extractor.py pipeline.

Input files (Fantasy & Paranormal subset):
  - goodreads_reviews_fantasy_paranormal.json.gz
  - goodreads_interactions_fantasy_paranormal.json.gz
  - goodreads_books_fantasy_paranormal.json.gz

Output files:
  - user_aggregated_reviews.csv   <- main input for LM_extractor
  - user_behavioral_features.csv  <- shelf/rating behavioral signals
  - user_genre_features.csv       <- per-user genre distribution
  - pipeline_stats.txt            <- summary stats for your records

Usage:
  python aggregate_reviews.py \
      --reviews  goodreads_reviews_fantasy_paranormal.json.gz \
      --interactions goodreads_interactions_fantasy_paranormal.json.gz \
      --books    goodreads_books_fantasy_paranormal.json.gz \
      --out_dir  ./aggregated \
      --min_reviews 10 \
      --max_tokens 512
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Aggregate Goodreads reviews per user.")
    p.add_argument("--reviews",      required=True,  help="Path to reviews .json.gz")
    p.add_argument("--interactions", required=True,  help="Path to interactions .json.gz")
    p.add_argument("--books",        required=True,  help="Path to books .json.gz")
    p.add_argument("--out_dir",      default="./aggregated", help="Output directory")
    p.add_argument("--min_reviews",  type=int, default=10,
                   help="Minimum number of reviews a user must have to be included (default: 10)")
    p.add_argument("--max_tokens",   type=int, default=512,
                   help="Approximate word budget per user blob (default: 512). "
                        "Matches BERT token limit in LM_extractor. Set 0 to disable truncation.")
    p.add_argument("--sample",       type=int, default=0,
                   help="If >0, only load this many lines from each file (useful for testing)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def iter_json_gz(path, sample=0):
    """Yield parsed JSON objects from a .json.gz file, one per line."""
    opener = gzip.open(path, "rt", encoding="utf-8")
    with opener as f:
        for i, line in enumerate(f):
            if sample and i >= sample:
                break
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def count_lines(path):
    """Count lines in a gzip file for tqdm progress bars."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_SPOILER_TAG = re.compile(r"\[.*?spoiler.*?\]", re.IGNORECASE)
_HTML_TAG    = re.compile(r"<[^>]+>")
_WHITESPACE  = re.compile(r"\s+")
_URL         = re.compile(r"https?://\S+|www\.\S+")

def clean_review_text(text: str) -> str:
    """Basic cleaning: strip spoiler tags, HTML, URLs, and excess whitespace."""
    if not text:
        return ""
    text = _SPOILER_TAG.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def truncate_to_word_budget(text: str, max_words: int) -> str:
    """
    Truncate text to approximately max_words words.
    BERT tokenizes into subword units so actual token count will be
    slightly higher (~1.2–1.4x words), but this keeps us in the right range.
    Set max_words=0 to skip truncation.
    """
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# Step 1 – Load books metadata (genre tags, title lookup)
# ---------------------------------------------------------------------------

def load_books(path, sample=0):
    """
    Returns:
        book_genres: dict[book_id -> list[str]]   genre tags from popular shelves
        book_titles: dict[book_id -> str]          book title lookup
    """
    print(f"\n[1/3] Loading books metadata from {path}")
    total = count_lines(path)

    book_genres = {}
    book_titles = {}

    for rec in tqdm(iter_json_gz(path, sample), total=total, unit="books"):
        bid = rec.get("book_id", "")
        if not bid:
            continue

        book_titles[bid] = rec.get("title", "")

        # popular_shelves is a list of {count, name} dicts
        shelves = rec.get("popular_shelves", [])
        genres = [s["name"] for s in shelves if isinstance(s, dict) and "name" in s]
        book_genres[bid] = genres[:10]  # keep top-10 shelf names per book

    print(f"    Loaded {len(book_titles):,} books.")
    return book_genres, book_titles


# ---------------------------------------------------------------------------
# Step 2 – Load interactions (shelf/rating behavior per user)
# ---------------------------------------------------------------------------

def load_interactions(path, sample=0):
    """
    Returns:
        interactions: dict[user_id -> list[dict]]
            Each dict has: book_id, is_read, rating, date_updated, shelf

    The interactions file schema:
        user_id, book_id, review_id, is_read, rating, date_updated, read_at
    """
    print(f"\n[2/3] Loading interactions from {path}")
    total = count_lines(path)

    interactions = defaultdict(list)

    for rec in tqdm(iter_json_gz(path, sample), total=total, unit="interactions"):
        uid = rec.get("user_id", "")
        bid = rec.get("book_id", "")
        if not uid or not bid:
            continue

        rating = rec.get("rating", 0)
        try:
            rating = int(rating)
        except (ValueError, TypeError):
            rating = 0

        interactions[uid].append({
            "book_id":      bid,
            "is_read":      bool(rec.get("is_read", False)),
            "rating":       rating,
            "date_updated": rec.get("date_updated", ""),
        })

    print(f"    Loaded interactions for {len(interactions):,} users.")
    return interactions


# ---------------------------------------------------------------------------
# Step 3 – Load and aggregate reviews
# ---------------------------------------------------------------------------

def load_and_aggregate_reviews(path, min_reviews=10, max_tokens=512, sample=0):
    """
    Streams reviews and aggregates per user.

    Returns:
        user_reviews: dict[user_id -> {
            'texts':        list[str]   cleaned review texts
            'review_count': int
            'rated_books':  list[book_id]
            'ratings':      list[int]
        }]
    """
    print(f"\n[3/3] Loading and aggregating reviews from {path}")
    total = count_lines(path)

    user_data = defaultdict(lambda: {
        "texts": [],
        "rated_books": [],
        "ratings": [],
    })

    skipped_empty = 0

    for rec in tqdm(iter_json_gz(path, sample), total=total, unit="reviews"):
        uid  = rec.get("user_id", "")
        bid  = rec.get("book_id", "")
        text = rec.get("review_text", "") or rec.get("body", "") or ""
        lang = rec.get("language_code", "")

        # Skip non-English and empty reviews
        # language_code is empty for many reviews; we keep those (likely English)
        if lang and lang not in ("", "en", "eng", "en-US", "en-GB"):
            skipped_empty += 1
            continue

        text = clean_review_text(text)
        if not text:
            skipped_empty += 1
            continue

        rating = rec.get("rating", 0)
        try:
            rating = int(rating)
        except (ValueError, TypeError):
            rating = 0

        user_data[uid]["texts"].append(text)
        if bid:
            user_data[uid]["rated_books"].append(bid)
        if rating > 0:
            user_data[uid]["ratings"].append(rating)

    print(f"    Raw user count (before filtering): {len(user_data):,}")
    print(f"    Skipped empty/non-English reviews: {skipped_empty:,}")

    # Filter by minimum review count
    filtered = {
        uid: v for uid, v in user_data.items()
        if len(v["texts"]) >= min_reviews
    }
    print(f"    Users with >= {min_reviews} reviews: {len(filtered):,}")

    # Build concatenated text blob per user
    for uid, v in filtered.items():
        blob = " ".join(v["texts"])
        blob = truncate_to_word_budget(blob, max_tokens)
        v["text_blob"]    = blob
        v["review_count"] = len(v["texts"])
        v["word_count"]   = len(blob.split())

    return filtered


# ---------------------------------------------------------------------------
# Step 4 – Build behavioral features from interactions
# ---------------------------------------------------------------------------

def build_behavioral_features(user_reviews, interactions):
    """
    Computes shelf/rating behavioral signals per user.

    Returns a list of dicts, one per user.
    """
    import statistics

    rows = []
    for uid, rv in user_reviews.items():
        user_ix = interactions.get(uid, [])

        # Rating stats
        ratings = [r for r in rv["ratings"] if r > 0]
        n_ratings = len(ratings)
        rating_mean   = round(statistics.mean(ratings), 3)   if ratings else None
        rating_stdev  = round(statistics.stdev(ratings), 3)  if len(ratings) > 1 else 0.0
        rating_min    = min(ratings)  if ratings else None
        rating_max    = max(ratings)  if ratings else None
        pct_5star     = round(ratings.count(5) / n_ratings, 3) if n_ratings else None
        pct_1star     = round(ratings.count(1) / n_ratings, 3) if n_ratings else None

        # Shelf/interaction stats
        n_shelved = len(user_ix)
        n_read    = sum(1 for i in user_ix if i["is_read"])
        read_rate = round(n_read / n_shelved, 3) if n_shelved else None

        rows.append({
            "user_id":       uid,
            "review_count":  rv["review_count"],
            "word_count":    rv["word_count"],
            "n_ratings":     n_ratings,
            "rating_mean":   rating_mean,
            "rating_stdev":  rating_stdev,
            "rating_min":    rating_min,
            "rating_max":    rating_max,
            "pct_5star":     pct_5star,
            "pct_1star":     pct_1star,
            "n_shelved":     n_shelved,
            "n_read":        n_read,
            "read_rate":     read_rate,
        })

    return rows


# ---------------------------------------------------------------------------
# Step 5 – Build genre features
# ---------------------------------------------------------------------------

# Broad genre buckets to reduce the long tail of shelf names
GENRE_KEYWORDS = {
    "fantasy":      ["fantasy", "magic", "fae", "dragon", "fairy", "wizard"],
    "paranormal":   ["paranormal", "vampire", "werewolf", "supernatural", "ghost", "witch"],
    "romance":      ["romance", "love", "contemporary-romance", "historical-romance"],
    "ya":           ["young-adult", "ya", "teen", "young adult"],
    "horror":       ["horror", "dark", "thriller", "scary"],
    "scifi":        ["sci-fi", "science-fiction", "dystopia", "space", "cyberpunk"],
    "mystery":      ["mystery", "crime", "detective", "cozy"],
    "literary":     ["literary", "classics", "literary-fiction", "general-fiction"],
    "nonfiction":   ["non-fiction", "nonfiction", "biography", "history", "science"],
    "dnf":          ["did-not-finish", "dnf", "abandoned"],
    "reread":       ["re-read", "reread", "favorites", "favourites"],
}

def classify_shelf(shelf_name: str) -> list[str]:
    """Map a shelf name to one or more genre buckets."""
    sl = shelf_name.lower().replace(" ", "-")
    return [bucket for bucket, kws in GENRE_KEYWORDS.items() if any(k in sl for k in kws)]


def build_genre_features(user_reviews, book_genres):
    """
    For each user, compute the fraction of their reviewed books
    that fall into each genre bucket.
    """
    rows = []
    for uid, rv in user_reviews.items():
        genre_counts = defaultdict(int)
        n_books = len(rv["rated_books"])

        for bid in rv["rated_books"]:
            shelves = book_genres.get(bid, [])
            seen_buckets = set()
            for shelf in shelves:
                for bucket in classify_shelf(shelf):
                    if bucket not in seen_buckets:
                        genre_counts[bucket] += 1
                        seen_buckets.add(bucket)

        row = {"user_id": uid, "n_reviewed_books": n_books}
        for bucket in GENRE_KEYWORDS:
            count = genre_counts.get(bucket, 0)
            row[f"genre_{bucket}"] = round(count / n_books, 4) if n_books else 0.0
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    start = datetime.now()

    # ---- Load ----
    book_genres, book_titles = load_books(args.books, sample=args.sample)
    interactions             = load_interactions(args.interactions, sample=args.sample)
    user_reviews             = load_and_aggregate_reviews(
        args.reviews,
        min_reviews=args.min_reviews,
        max_tokens=args.max_tokens,
        sample=args.sample,
    )

    # ---- Build outputs ----
    print("\n[Building output dataframes...]")

    # 1. Main aggregated review CSV (input to LM_extractor)
    review_rows = [
        {"user_id": uid, "text": v["text_blob"], "review_count": v["review_count"], "word_count": v["word_count"]}
        for uid, v in user_reviews.items()
    ]
    df_reviews = pd.DataFrame(review_rows).sort_values("user_id").reset_index(drop=True)

    # 2. Behavioral features
    behavioral_rows = build_behavioral_features(user_reviews, interactions)
    df_behavioral   = pd.DataFrame(behavioral_rows).sort_values("user_id").reset_index(drop=True)

    # 3. Genre features
    genre_rows = build_genre_features(user_reviews, book_genres)
    df_genre   = pd.DataFrame(genre_rows).sort_values("user_id").reset_index(drop=True)

    # ---- Save ----
    reviews_path     = os.path.join(args.out_dir, "user_aggregated_reviews.csv")
    behavioral_path  = os.path.join(args.out_dir, "user_behavioral_features.csv")
    genre_path       = os.path.join(args.out_dir, "user_genre_features.csv")
    stats_path       = os.path.join(args.out_dir, "pipeline_stats.txt")

    df_reviews.to_csv(reviews_path,    index=False)
    df_behavioral.to_csv(behavioral_path, index=False)
    df_genre.to_csv(genre_path,        index=False)

    elapsed = (datetime.now() - start).seconds
    n_users = len(df_reviews)

    stats = f"""
Pipeline run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Elapsed: {elapsed}s

Settings
--------
min_reviews : {args.min_reviews}
max_tokens  : {args.max_tokens}
sample      : {args.sample if args.sample else 'full dataset'}

Outputs
-------
Users retained          : {n_users:,}
Books loaded            : {len(book_titles):,}
Users with interactions : {len(interactions):,}

Review text stats
-----------------
Mean review count / user : {df_reviews['review_count'].mean():.1f}
Median review count      : {df_reviews['review_count'].median():.0f}
Mean word count / blob   : {df_reviews['word_count'].mean():.1f}

Files written
-------------
{reviews_path}
{behavioral_path}
{genre_path}
""".strip()

    with open(stats_path, "w") as f:
        f.write(stats)

    print(f"\n{'='*55}")
    print(stats)
    print(f"{'='*55}")
    print(f"\nDone. Outputs written to: {args.out_dir}/")
    print(f"\nNext step:")
    print(f"  python LM_extractor.py -dataset_type custom \\")
    print(f"      -input_csv {reviews_path} \\")
    print(f"      -text_col text -token_length {args.max_tokens} \\")
    print(f"      -batch_size 32 -embed bert-base -op_dir pkl_data")


if __name__ == "__main__":
    main()
