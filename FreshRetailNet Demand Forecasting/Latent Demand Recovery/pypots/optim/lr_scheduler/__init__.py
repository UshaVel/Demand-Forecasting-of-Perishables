"""
Learning rate schedulers. Trimmed: only the base class is kept, since Adam
is used here without a scheduler (the original also exposes Lambda/Multiplicative/
Step/MultiStep/Constant/Exponential/Linear LR schedulers).
"""

from .base import LRScheduler

__all__ = [
    "LRScheduler",
]
