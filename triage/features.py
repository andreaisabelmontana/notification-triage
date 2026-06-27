"""Hand-crafted lexical / structural signals for a notification.

These complement the learned TF-IDF + LSA semantic vector. None of this is a
large language model: it is regex and lexicon counting over the raw text. The
point is to capture cheap, interpretable urgency cues (the words "URGENT", a
deadline, a 2FA-style code) that the bag-of-words representation alone tends to
under-weight on a small training set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words that, in a notification, tend to signal something needs attention now.
URGENCY_LEXICON = frozenset(
    {
        "urgent",
        "immediately",
        "now",
        "asap",
        "critical",
        "emergency",
        "alert",
        "important",
        "action",
        "required",
        "expires",
        "expiring",
        "deadline",
        "overdue",
        "final",
        "warning",
        "failed",
        "failure",
        "down",
        "outage",
        "suspended",
        "fraud",
        "suspicious",
        "blocked",
        "respond",
        "confirm",
        "verify",
        "today",
        "tonight",
        "evacuate",
    }
)

# Senders ranked roughly by how much their messages tend to matter.
SENDER_WEIGHTS = {
    "security": 1.0,
    "ops": 1.0,
    "billing": 0.8,
    "calendar": 0.6,
    "personal": 0.7,
    "travel": 0.6,
    "orders": 0.5,
    "work": 0.5,
    "device": 0.3,
    "social": 0.15,
    "marketing": 0.0,
}

_WORD_RE = re.compile(r"[a-z']+")
# A digit run of length 4-8 — verification codes, invoice / flight numbers.
_CODE_RE = re.compile(r"\b\d{4,8}\b")
# Money amounts like $4,200 or $50.
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*")
# Time / deadline mentions.
_TIME_RE = re.compile(
    r"\b(\d{1,2}\s?(?:am|pm)|\d{1,2}:\d{2}|"
    r"today|tonight|tomorrow|now|minutes?|hours?|days?|"
    r"deadline|due|expires?|expiring)\b"
)

# Stable, documented order of the engineered features.
FEATURE_NAMES = (
    "urgency_hits",
    "urgency_density",
    "caps_ratio",
    "exclaim_count",
    "has_code",
    "has_money",
    "time_mentions",
    "sender_weight",
    "length_words",
)


@dataclass(frozen=True)
class LexicalFeatures:
    urgency_hits: float
    urgency_density: float
    caps_ratio: float
    exclaim_count: float
    has_code: float
    has_money: float
    time_mentions: float
    sender_weight: float
    length_words: float
    names: tuple = field(default=FEATURE_NAMES, repr=False)

    def as_vector(self) -> list:
        return [getattr(self, n) for n in FEATURE_NAMES]


def caps_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are upper-case (shouting)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def lexical_features(text: str, sender: str = "") -> LexicalFeatures:
    """Extract interpretable urgency signals from one notification."""
    lower = text.lower()
    words = _WORD_RE.findall(lower)
    n_words = max(len(words), 1)

    hits = sum(1 for w in words if w in URGENCY_LEXICON)

    return LexicalFeatures(
        urgency_hits=float(hits),
        urgency_density=hits / n_words,
        caps_ratio=caps_ratio(text),
        exclaim_count=float(text.count("!")),
        has_code=1.0 if _CODE_RE.search(text) else 0.0,
        has_money=1.0 if _MONEY_RE.search(text) else 0.0,
        time_mentions=float(len(_TIME_RE.findall(lower))),
        sender_weight=SENDER_WEIGHTS.get(sender.strip().lower(), 0.4),
        length_words=float(len(words)),
    )
