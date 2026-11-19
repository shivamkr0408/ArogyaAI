"""
Per-symptom explanation of a Naive Bayes prediction.

Naive Bayes is one of the few classifiers whose decision decomposes exactly,
so this is a real attribution rather than a post-hoc approximation like LIME
or SHAP would be. Both deployed variants decompose, but differently:

**Multinomial NB**

    log P(c | x)  =  log P(c)  +  SUM_i  x_i * log P(f_i | c)   + const

Every *present* symptom contributes ``feature_log_prob_[c][i]``; absent
symptoms contribute nothing at all.

**Bernoulli NB** (the default)

    log P(c | x)  =  log P(c)  +  SUM_i [ x_i*log t_ci + (1-x_i)*log(1-t_ci) ]

which sklearn evaluates as ``x @ (log t - log(1-t)).T + sum(log(1-t))``. So a
present symptom contributes its **log-odds** ``log t_ci - log(1-t_ci)``, and
a symptom's *absence* is itself evidence. That is why the NegEx/ConText layer
matters to the model and not just to the transcript: when a user says "no
cough", that absence genuinely shifts the posterior, and this module can show
by how much.

Either way the itemized numbers sum back to the model's own log-score, which
`tests/test_explain.py` asserts.

Two views are produced:

* **Support** -- how much each reported symptom contributed to the top
  condition.
* **Discrimination** -- the log-odds of each symptom for the top condition
  *versus the runner-up*, which is what actually explains why one was
  chosen over the other. A symptom common to both conditions has high
  support but near-zero discrimination, and saying so is more honest than
  listing it as a reason.
"""

import logging
import math
from typing import Optional

import numpy as np

from . import predictor

logger = logging.getLogger(__name__)


def _vector(symptom_flags: dict, feature_order: list[str]) -> np.ndarray:
    vector = np.zeros(len(feature_order))
    for i, symptom in enumerate(feature_order):
        if symptom_flags.get(symptom):
            vector[i] = 1.0
    return vector


def _weights(model, class_index: int, n_features: int):
    """
    Per-feature weight for a *present* symptom, and the model family.

    For Bernoulli this is the log-odds (so absence carries weight too); for
    Multinomial it is the plain log-likelihood.
    """
    feature_log_prob = np.asarray(model.feature_log_prob_[class_index])
    family = type(model).__name__

    if family == "BernoulliNB":
        # log(1 - theta), computed stably from log(theta).
        with np.errstate(divide="ignore"):
            neg_log_prob = np.log(np.clip(1.0 - np.exp(feature_log_prob), 1e-12, None))
        return feature_log_prob - neg_log_prob, neg_log_prob, family

    return feature_log_prob, None, family


def explain(
    symptom_flags: dict,
    top_disease: Optional[str] = None,
    runner_up: Optional[str] = None,
    max_missing: int = 4,
    negated_symptoms: Optional[list[str]] = None,
) -> Optional[dict]:
    """
    Decompose the prediction for `top_disease` into per-symptom evidence.

    Returns None when the model is not loaded or carries no learned
    log-probabilities (a non-Bayes estimator swapped in, for instance).
    """
    model = predictor.get_model()
    feature_order = predictor.get_feature_order()
    classes = predictor.get_label_classes()

    if model is None or not feature_order or not classes:
        return None
    feature_log_prob = getattr(model, "feature_log_prob_", None)
    class_log_prior = getattr(model, "class_log_prior_", None)
    if feature_log_prob is None:
        return None

    present = [s for s in feature_order if symptom_flags.get(s)]
    if not present:
        return None

    # Resolve the two classes being compared.
    predictions = predictor.predict(symptom_flags)
    if top_disease is None:
        if not predictions:
            return None
        top_disease = predictions[0]["disease"]
    if runner_up is None and len(predictions) > 1:
        runner_up = predictions[1]["disease"]

    try:
        top_idx = classes.index(top_disease)
    except ValueError:
        return None
    runner_idx = None
    if runner_up is not None and runner_up in classes:
        runner_idx = classes.index(runner_up)

    index_of = {name: i for i, name in enumerate(feature_order)}
    n_features = len(feature_order)

    top_weights, top_neg, family = _weights(model, top_idx, n_features)
    runner_weights = None
    if runner_idx is not None:
        runner_weights, _, _ = _weights(model, runner_idx, n_features)

    contributions = []
    for symptom in present:
        i = index_of[symptom]
        support = float(top_weights[i])
        discrimination = None
        if runner_weights is not None:
            discrimination = support - float(runner_weights[i])
        contributions.append(
            {
                "symptom": symptom,
                "label": symptom.replace("_", " "),
                "log_likelihood": round(support, 4),
                "discrimination": (
                    round(discrimination, 4) if discrimination is not None else None
                ),
            }
        )

    # Share of total evidence, so the UI can draw comparable bars. Weights are
    # shifted to be non-negative first: Bernoulli log-odds can be either sign,
    # and a share has to be taken over magnitudes that are comparable.
    raw = [c["log_likelihood"] for c in contributions]
    floor = min(raw) if raw else 0.0
    shifted = [(v - floor) + 1e-6 for v in raw]
    total = sum(shifted) or 1.0
    for contribution, value in zip(contributions, shifted):
        contribution["support_share"] = round(value / total, 4)

    contributions.sort(key=lambda c: -c["log_likelihood"])

    # Bernoulli only: symptoms the user explicitly *denied*, and how much
    # that denial moved the score for this condition. This is the NegEx
    # output feeding the model as real evidence.
    denial_evidence = []
    if family == "BernoulliNB" and top_neg is not None and negated_symptoms:
        for symptom in negated_symptoms:
            i = index_of.get(symptom)
            if i is None or symptom_flags.get(symptom):
                continue
            effect = float(top_neg[i])
            against = None
            if runner_idx is not None:
                _, runner_neg, _ = _weights(model, runner_idx, n_features)
                if runner_neg is not None:
                    against = effect - float(runner_neg[i])
            denial_evidence.append(
                {
                    "symptom": symptom,
                    "label": symptom.replace("_", " "),
                    "log_absence_likelihood": round(effect, 4),
                    "discrimination": round(against, 4) if against is not None else None,
                }
            )
        denial_evidence.sort(key=lambda d: d["discrimination"] or 0, reverse=True)

    discriminating = [c for c in contributions if c["discrimination"] is not None]
    discriminating.sort(key=lambda c: -(c["discrimination"] or 0))

    # Symptoms the model expects for this condition that were not reported --
    # a genuinely useful prompt for the user, and a check on overconfidence.
    missing = []
    for symptom in feature_order:
        if symptom_flags.get(symptom):
            continue
        i = index_of[symptom]
        missing.append((symptom, float(top_weights[i])))
    missing.sort(key=lambda kv: -kv[1])
    missing_out = [
        {"symptom": s, "label": s.replace("_", " "), "log_likelihood": round(v, 4)}
        for s, v in missing[:max_missing]
    ]

    prior = (
        round(float(class_log_prior[top_idx]), 4)
        if class_log_prior is not None
        else None
    )

    if family == "BernoulliNB":
        weight_meaning = (
            "log-odds of the symptom under this condition, log P(s|c) - "
            "log P(not s|c). Bernoulli NB also scores symptom *absence*, so a "
            "denied symptom is evidence too."
        )
    else:
        weight_meaning = (
            "log-likelihood log P(s|c). Multinomial NB scores only the "
            "symptoms that are present; absences contribute nothing."
        )

    return {
        "condition": top_disease,
        "compared_with": runner_up,
        "model": family,
        "weight_meaning": weight_meaning,
        "log_prior": prior,
        "reported_symptom_count": len(present),
        "contributions": contributions,
        "top_discriminating": discriminating[:5],
        "least_discriminating": (
            [c for c in reversed(discriminating[-3:])] if len(discriminating) > 3 else []
        ),
        "denial_evidence": denial_evidence,
        "expected_but_not_reported": missing_out,
        "note": (
            "Naive Bayes decomposes exactly: each reported symptom's weight "
            "below sums into the model's score for this condition. "
            "'Discrimination' is that weight for "
            + str(top_disease)
            + " minus the same weight for "
            + str(runner_up or "the next condition")
            + "; values near zero mean the symptom does not separate them."
        ),
    }


def summarize(explanation: Optional[dict], limit: int = 3) -> str:
    """One sentence naming the symptoms that actually drove the decision."""
    if not explanation:
        return ""
    drivers = explanation.get("top_discriminating") or explanation.get("contributions")
    if not drivers:
        return ""
    names = [d["label"] for d in drivers[:limit]]
    if len(names) == 1:
        joined = names[0]
    else:
        joined = ", ".join(names[:-1]) + " and " + names[-1]
    compared = explanation.get("compared_with")
    if compared:
        return (
            f"{joined.capitalize()} weighed most heavily towards "
            f"{explanation['condition']} rather than {compared}."
        )
    return f"{joined.capitalize()} weighed most heavily towards {explanation['condition']}."
