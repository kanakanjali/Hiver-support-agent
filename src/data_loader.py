"""
Data Loader — download, clean, and filter the Customer Support on Twitter dataset.

Handles:
  - Downloading via kagglehub (or loading a local CSV)
  - Filtering to a single brand (default: AppleSupport)
  - Basic text cleaning (removing URLs, normalising whitespace)
  - Random subsampling for tractable experiments
"""

import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


# ── Download / Load ────────────────────────────────────────────────────────────

def download_dataset(target_dir: Optional[Path] = None) -> Path:
    """Download dataset via kagglehub; returns path to the CSV directory."""
    target_dir = target_dir or config.RAW_DATA_DIR
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "thoughtvector/customer-support-on-twitter"
        )
        logger.info("Dataset downloaded to %s", path)
        return Path(path)
    except Exception as e:
        logger.error("kagglehub download failed: %s", e)
        logger.info("Falling back to local data in %s", target_dir)
        return target_dir


def load_dataset(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the tweets CSV into a DataFrame.

    Looks for 'twcs.csv' (the standard filename) in *data_path*.
    If *data_path* is None, attempts download first.
    """
    if data_path is None:
        data_path = download_dataset()

    data_path = Path(data_path)

    # data_path might be a directory or a file
    if data_path.is_dir():
        csv_candidates = list(data_path.glob("*.csv"))
        if not csv_candidates:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        csv_path = csv_candidates[0]  # typically twcs.csv
    else:
        csv_path = data_path

    logger.info("Loading %s …", csv_path)
    df = pd.read_csv(csv_path, dtype={"tweet_id": str, "in_response_to_tweet_id": str})

    # Normalise column names (some versions differ)
    df.columns = [c.strip().lower() for c in df.columns]

    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
    return df


# ── Filtering ──────────────────────────────────────────────────────────────────

def filter_brand(df: pd.DataFrame, brand: str = config.BRAND) -> pd.DataFrame:
    """
    Keep only rows that are *from* the brand or *to* the brand.

    The brand's outbound replies have ``author_id == brand`` and ``inbound == False``.
    Customer inbound messages are identified by having a response from the brand.
    """
    # Outbound brand replies
    brand_mask = df["author_id"].astype(str).str.lower() == brand.lower()
    brand_tweet_ids = set(df.loc[brand_mask, "tweet_id"].dropna().astype(str))

    # Inbound messages that the brand replied to
    inbound_mask = df["tweet_id"].astype(str).isin(
        df.loc[brand_mask, "in_response_to_tweet_id"].dropna().astype(str)
    )

    # Also keep inbound messages whose response is a brand tweet
    if "response_tweet_id" in df.columns:
        response_ids = (
            df["response_tweet_id"]
            .dropna()
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )
        inbound_via_response = df["tweet_id"].astype(str).isin(
            df.loc[
                df["tweet_id"].astype(str).isin(response_ids) & brand_mask,
                "in_response_to_tweet_id",
            ]
            .dropna()
            .astype(str)
        )
        combined = brand_mask | inbound_mask | inbound_via_response
    else:
        combined = brand_mask | inbound_mask

    filtered = df.loc[combined].copy()
    logger.info(
        "Filtered to brand '%s': %d → %d rows", brand, len(df), len(filtered)
    )
    return filtered


# ── Text Cleaning ──────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove URLs, @mentions, and normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def add_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'clean_text' column to the DataFrame."""
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    return df


# ── Subsampling ────────────────────────────────────────────────────────────────

def subsample(
    df: pd.DataFrame,
    max_rows: int = config.MAX_CONVERSATIONS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Deterministic random subsample."""
    if len(df) <= max_rows:
        return df
    sampled = df.sample(n=max_rows, random_state=seed)
    logger.info("Subsampled %d → %d rows", len(df), len(sampled))
    return sampled


# ── Top-level convenience ─────────────────────────────────────────────────────

def prepare_brand_data(
    brand: str = config.BRAND,
    max_rows: int = config.MAX_CONVERSATIONS,
    data_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Full pipeline: load → filter → clean → subsample."""
    df = load_dataset(data_path)
    df = filter_brand(df, brand)
    df = add_clean_text(df)
    df = subsample(df, max_rows)
    return df
