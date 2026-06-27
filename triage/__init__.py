"""notification-triage: rank notifications by priority from their content.

Semantic analysis = TF-IDF + Latent Semantic Analysis (truncated SVD)
embeddings plus a lexical urgency layer, fed to a logistic-regression
classifier. Not a large language model.
"""

from .features import LexicalFeatures, lexical_features
from .model import (
    Notification,
    PriorityModel,
    RankedNotification,
    load_dataset,
    topic_boost,
    train_default,
)

__all__ = [
    "LexicalFeatures",
    "lexical_features",
    "Notification",
    "PriorityModel",
    "RankedNotification",
    "load_dataset",
    "topic_boost",
    "train_default",
]

__version__ = "0.1.0"
