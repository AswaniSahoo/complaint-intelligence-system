"""Shared fixtures for all test modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def sample_df():
    """Load a small slice of the processed complaints for testing."""
    path = DATA_DIR / "processed_complaints.csv"
    if not path.exists():
        pytest.skip("processed_complaints.csv not found — run pipeline first")
    df = pd.read_csv(path, nrows=500)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@pytest.fixture(scope="session")
def sample_embeddings():
    """Load a small slice of embeddings matching sample_df."""
    path = DATA_DIR / "embeddings.npy"
    if not path.exists():
        pytest.skip("embeddings.npy not found — run pipeline first")
    return np.load(path)[:500]


@pytest.fixture(scope="session")
def tiny_df():
    """Minimal synthetic DataFrame for unit tests (no disk dependency)."""
    return pd.DataFrame({
        "complaint_text": [
            "My credit card was charged twice for the same purchase.",
            "I never received my mortgage statement this month.",
            "The bank closed my account without any notice.",
            "Late fees were added even though I paid on time.",
            "I found unauthorized transactions on my debit card.",
        ],
        "clean_text": [
            "credit card charged twice same purchase",
            "never received mortgage statement month",
            "bank closed account without notice",
            "late fees added paid on time",
            "unauthorized transactions debit card",
        ],
        "product": [
            "Credit card",
            "Mortgage",
            "Checking or savings account",
            "Credit card",
            "Checking or savings account",
        ],
        "issue": [
            "Billing dispute",
            "Missing statement",
            "Account closure",
            "Late fees",
            "Unauthorized transactions",
        ],
        "date": pd.to_datetime([
            "2024-01-15", "2024-02-20", "2024-03-10",
            "2024-04-05", "2024-05-18",
        ]),
        "cluster": [0, 1, 2, 0, 2],
    })
