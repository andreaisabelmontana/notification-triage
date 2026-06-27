"""Tests for the notification triage model.

Run:  python -m pytest -q
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from triage import Notification, load_dataset, topic_boost, train_default
from triage.features import lexical_features
from triage.model import PriorityModel


@pytest.fixture(scope="module")
def model():
    return train_default()


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


# --------------------------------------------------------------------------- #
# Behavioural: urgent beats noise                                             #
# --------------------------------------------------------------------------- #


def test_urgent_outranks_newsletter(model):
    urgent = Notification(
        "URGENT: production down, respond now", "ops"
    )
    newsletter = Notification(
        "Weekly newsletter: 5 articles we think you'll enjoy", "marketing"
    )
    ranked = model.rank([newsletter, urgent])
    top = ranked[0].notification
    assert top.text == urgent.text
    # And the scores must actually separate them.
    by_text = {r.notification.text: r.score for r in ranked}
    assert by_text[urgent.text] > by_text[newsletter.text]


def test_security_code_outranks_social_like(model):
    code = Notification("Your verification code is 481920", "security")
    like = Notification("Maria liked your photo", "social")
    ranked = model.rank([like, code])
    assert ranked[0].notification.text == code.text


# --------------------------------------------------------------------------- #
# Held-out accuracy above a floor                                             #
# --------------------------------------------------------------------------- #


def test_held_out_accuracy_above_floor(dataset):
    texts, senders, labels = dataset
    idx = np.arange(len(texts))
    tr, te = train_test_split(
        idx, test_size=0.25, random_state=0, stratify=labels
    )
    m = PriorityModel()
    m.fit(
        [texts[i] for i in tr],
        [senders[i] for i in tr],
        [labels[i] for i in tr],
    )
    preds = m.predict_priority(
        [texts[i] for i in te], [senders[i] for i in te]
    )
    truth = [labels[i] for i in te]
    acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)

    # Majority-class baseline on this 3-class set is ~0.42; require well above.
    assert acc >= 0.70, f"held-out accuracy too low: {acc:.3f}"


# --------------------------------------------------------------------------- #
# Ranking is a stable total order                                             #
# --------------------------------------------------------------------------- #


def test_ranking_is_stable_total_order(model):
    batch = [
        Notification("URGENT: server down, act now", "ops"),
        Notification("Your code is 123456", "security"),
        Notification("Team standup at 9:30am", "calendar"),
        Notification("You have 3 new followers this week", "social"),
        Notification("50% off this weekend only", "marketing"),
    ]
    ranked = model.rank(batch)

    # Every item appears exactly once -> it is a permutation (total order).
    assert sorted(r.rank for r in ranked) == [1, 2, 3, 4, 5]
    assert len({r.notification.text for r in ranked}) == len(batch)

    # Scores are monotonically non-increasing along the ranking.
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)

    # Deterministic: same input -> same order, twice.
    again = model.rank(batch)
    assert [r.notification.text for r in ranked] == [
        r.notification.text for r in again
    ]


def test_ties_broken_by_original_index(model):
    # Two identical messages must keep a deterministic, index-based order.
    a = Notification("Team standup at 9:30am", "calendar")
    b = Notification("Team standup at 9:30am", "calendar")
    ranked = model.rank([a, b])
    assert [r.rank for r in ranked] == [1, 2]


# --------------------------------------------------------------------------- #
# Semantic topic boost                                                         #
# --------------------------------------------------------------------------- #


def test_topic_similarity_boosts_score(model):
    texts = [
        "Your flight has been rescheduled to a later time",
        "Your weekly newsletter is ready to read",
    ]
    senders = ["travel", "marketing"]
    topics = ["flight travel itinerary airport"]

    base = model.priority_score(texts, senders)
    boosted = topic_boost(model, texts, senders, topics, weight=0.4)

    # The flight message is semantically close to the topic; it must gain more
    # than the unrelated newsletter.
    flight_gain = boosted[0] - base[0]
    news_gain = boosted[1] - base[1]
    assert flight_gain > news_gain
    assert flight_gain > 0


# --------------------------------------------------------------------------- #
# Lexical feature sanity                                                       #
# --------------------------------------------------------------------------- #


def test_lexical_urgency_signals():
    urgent = lexical_features("URGENT! Respond immediately now", "ops")
    calm = lexical_features("a blog you follow published a new post", "social")
    assert urgent.urgency_hits > calm.urgency_hits
    assert urgent.caps_ratio > calm.caps_ratio
    assert urgent.exclaim_count >= 1
    assert urgent.sender_weight > calm.sender_weight


def test_code_and_money_detection():
    f = lexical_features("Your code 481920 and a charge of $4,200", "billing")
    assert f.has_code == 1.0
    assert f.has_money == 1.0
