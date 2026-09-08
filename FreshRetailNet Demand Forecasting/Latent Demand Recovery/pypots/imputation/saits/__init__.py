"""
Trimmed: only DatasetForSAITS is kept, because TimesNet's dataset class reuses
SAITS's masking strategy (the original also exposes the full SAITS model here).
"""

from .data import DatasetForSAITS

__all__ = [
    "DatasetForSAITS",
]
