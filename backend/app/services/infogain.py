"""
Next-best-question selection by expected information gain.

The old intake picked the next symptom to ask about from a static
"discriminating symptoms" table. That asks a reasonable question, but not
necessarily the most useful one: a symptom shared by the top three
candidates cannot separate them however characteristic it is.

This chooses the question that is expected to reduce uncertainty most.
Writing H for Shannon entropy of the posterior over conditions:

    IG(s) = H(C) - [ P(s=yes) * H(C | s=yes) + P(s=no) * H(C | s=no) ]

Both branches are evaluated by actually re-running the Naive Bayes update
with the symptom set to present and to absent, so the number reported is
the model's own expectation rather than a proxy for it. P(s=yes) is the
posterior-weighted prevalence, sum_c P(c) * P(s=1|c), read straight off the
fitted estimator's feature_log_prob_.

Cost is O(|S| * |C|) -- about 102 x 41 here -- which is well under a
millisecond, so it runs inline on every turn.

The gain is returned alongside the question so the interface can answer
"why are you asking me this?" with the real number.
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Stop asking once the differential is this clear.
CONFIDENT_ENOUGH = 0.70
# Hard ceiling on questions, so the consultation cannot run forever.
MAX_QUESTIONS = 5
# A question expected to move things less than this is not worth asking.
MIN_USEFUL_GAIN = 0.01


def _entropy(p: np.ndarray) -> float:
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def _posterior(model, vector: np.ndarray) -> np.ndarray:
    lp = model.predict_log_proba(vector.reshape(1, -1))[0]
    lp = lp - lp.max()
    e = np.exp(lp)
    return e / e.sum()


def rank_questions(
    model,
    feature_order: list[str],
    present: set[str],
    absent: set[str],
    already_asked: set[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Rank unasked symptoms by expected information gain.

    `present` and `absent` are what the user has told us; `already_asked`
    prevents repeating a question whose answer was "not sure".
    """
    if model is None or not feature_order:
        return []

    try:
        # log P(x_i = 1 | c) for every class/feature pair.
        feat_log_prob = np.asarray(model.feature_log_prob_, dtype=float)
    except AttributeError:
        logger.debug("estimator exposes no feature_log_prob_; skipping info gain")
        return []

    index = {s: i for i, s in enumerate(feature_order)}
    base = np.zeros(len(feature_order))
    for s in present:
        if s in index:
            base[index[s]] = 1.0

    posterior = _posterior(model, base)
    h_now = _entropy(posterior)
    if h_now <= 0:
        return []

    p_present = np.exp(feat_log_prob)          # (n_classes, n_features)
    skip = present | absent | already_asked

    scored = []
    for s, j in index.items():
        if s in skip:
            continue

        # Posterior-weighted chance the user says yes.
        p_yes = float((posterior * p_present[:, j]).sum())
        p_yes = min(max(p_yes, 1e-6), 1 - 1e-6)

        v_yes = base.copy()
        v_yes[j] = 1.0
        h_yes = _entropy(_posterior(model, v_yes))

        # "No" is represented by leaving the feature at zero; for Bernoulli
        # NB that is a real observation, not a missing value, which is the
        # reason this variant is deployed.
        v_no = base.copy()
        v_no[j] = 0.0
        h_no = _entropy(_posterior(model, v_no))

        expected = p_yes * h_yes + (1 - p_yes) * h_no
        gain = h_now - expected
        if gain > MIN_USEFUL_GAIN:
            scored.append({
                "symptom": s,
                "information_gain": round(float(gain), 4),
                "p_yes": round(p_yes, 4),
                "entropy_now": round(h_now, 4),
                "entropy_if_yes": round(float(h_yes), 4),
                "entropy_if_no": round(float(h_no), 4),
            })

    scored.sort(key=lambda d: -d["information_gain"])
    return scored[:top_n]


def best_question(
    model,
    feature_order: list[str],
    present: set[str],
    absent: set[str],
    already_asked: set[str],
) -> Optional[dict]:
    ranked = rank_questions(model, feature_order, present, absent, already_asked, top_n=1)
    return ranked[0] if ranked else None


def should_stop(top_probability: float, asked: int) -> tuple[bool, str]:
    """Stop when the differential is clear enough, or we have asked enough."""
    if top_probability >= CONFIDENT_ENOUGH:
        return True, f"top candidate reached {top_probability:.0%}"
    if asked >= MAX_QUESTIONS:
        return True, f"asked the maximum of {MAX_QUESTIONS} questions"
    return False, ""


def explain(candidate: dict) -> str:
    """One sentence for the 'why am I asking this?' tooltip."""
    if not candidate:
        return ""
    pct = candidate["p_yes"] * 100
    return (
        f"Expected information gain {candidate['information_gain']:.2f} bits. "
        f"Uncertainty across the conditions is {candidate['entropy_now']:.2f} bits now; "
        f"a yes would take it to {candidate['entropy_if_yes']:.2f} and a no to "
        f"{candidate['entropy_if_no']:.2f}. Based on the current differential there is "
        f"roughly a {pct:.0f}% chance you say yes, and this is the question that "
        f"separates the leading candidates most."
    )
