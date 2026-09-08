"""
Transformer building blocks. Trimmed: only the embedding module is kept, since
TimesNet only needs DataEmbedding (the original also exposes attention,
encoder/decoder, and layer modules used by other models).
"""

from .embedding import DataEmbedding, PositionalEncoding

__all__ = [
    "DataEmbedding",
    "PositionalEncoding",
]
