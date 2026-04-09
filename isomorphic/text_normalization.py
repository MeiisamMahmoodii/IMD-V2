from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Normalize text before downstream distance calculations."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

