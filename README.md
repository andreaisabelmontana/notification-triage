# notification-triage

Ranks notifications by priority from their content, so the message that matters
reaches you first instead of whatever arrived last.

It reads each notification's text, scores it on a continuous priority scale, and
returns a ranked inbox. A clearly urgent alert ("URGENT: production down,
respond now") rises to the top; a newsletter or a "liked your photo" sinks.

## What "semantic analysis" means here (honest version)

This is **not** a large language model. "Semantic" means:

- **TF-IDF + LSA embeddings** — each message is turned into a TF-IDF vector
  (uni/bi-grams) and compressed with **truncated SVD** (Latent Semantic
  Analysis) into a dense ~40-dim vector. This is what lets it tell
  *"your account was accessed from a new device"* (urgent) from
  *"new devices are on sale"* (noise) despite the shared words.
- **Lexical urgency signals** — cheap, interpretable cues added alongside the
  embedding: urgency-lexicon hits (`urgent`, `immediately`, `expires`…),
  ALL-CAPS ratio, exclamation marks, presence of a 4–8 digit code, money
  amounts, deadline/time mentions, and a sender-category weight.

The two feature blocks are concatenated and fed to a logistic-regression
classifier. The whole model trains in under a second on CPU and is fully
inspectable — no external services, no API keys.

## Model

```
text ──TF-IDF (1,2-grams)──► sparse ──truncated SVD (LSA)──► dense embedding ┐
text ──lexical signals (urgency lexicon, caps, code, $, deadlines, sender)──► ┤─► LogisticRegression ─► P(low/med/high)
                                                                              ┘
```

The class probabilities are collapsed into a single **priority score in [0,1]**
(probability-weighted average of tier values: low=0, medium=0.5, high=1), and
the inbox is sorted by that score. Ties break by arrival index, so the ranking
is always a strict total order.

`topic_boost()` additionally nudges a message's score up by its cosine
similarity (in LSA space) to a user's "important topics" — genuinely semantic,
not keyword overlap.

## Training data

`data/notifications.csv` — a hand-built seed set of **65 notifications**
labelled `high` / `medium` / `low` (20 / 20 / 25), spanning security alerts,
ops outages, billing, calendar, orders, social, and marketing. Created and
committed in this repo; small by design and easy to extend.

## Run it

```bash
pip install -r requirements.txt
python demo.py            # ranked demo inbox
python -m pytest -q       # tests
```

## Real ranked example

`python demo.py` on a mixed batch (output is real, not illustrative):

```
Ranked inbox (highest priority first)

 #  score  tier     message
------------------------------------------------------------------------------
 1  1.000  high     URGENT: production is down, customers cannot chec...
 2  0.999  high     Security alert: your account was accessed from a ...
 3  0.532  medium   Your package will be delivered today between 2pm ...
 4  0.509  medium   Your password reset link expires in 30 minutes
 5  0.497  medium   Reminder: dentist appointment tomorrow at 10am
 6  0.046  low      Maria liked your photo
 7  0.021  low      Weekly newsletter: 5 articles we think you'll enj...
 8  0.011  low      50% off everything this weekend only, shop the sale
```

## Held-out accuracy

On a 25% stratified hold-out (48 train / 17 test, `random_state=0`):

```
held-out: 17 test / 48 train -> accuracy 0.941 (16/17)
```

The majority-class baseline on this 3-class set is ~0.42, so the model is well
above floor. (The test suite asserts ≥ 0.70 to stay robust across splits.)

## Tests

`tests/test_triage.py` — 8 tests, all passing:

```
........                                                                 [100%]
8 passed in 2.84s
```

They check that an urgent message outranks a newsletter, that held-out accuracy
clears the floor, that the ranking is a stable strict total order (with
deterministic tie-breaking), that semantic similarity to user topics boosts
score, and that the lexical signals fire correctly.

## Layout

```
triage/
  features.py   lexical urgency signals (lexicon, caps, codes, deadlines, sender)
  model.py      TF-IDF + LSA embedding + classifier, scoring & ranking
data/
  notifications.csv   65 labelled seed notifications
demo.py         ranked demo inbox
tests/          pytest suite
```

## License

MIT — see [LICENSE](LICENSE).
