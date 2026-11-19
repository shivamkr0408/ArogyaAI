"""
Probability calibration for the deployed classifier.

Why this exists: the model was reporting things like "GERD 76%" from three
common symptoms. Naive Bayes multiplies per-symptom likelihoods under an
independence assumption that real symptoms violate -- correlated symptoms
get counted as if they were independent pieces of evidence, so the winning
class is pushed towards 1.0. The *ranking* is still informative; the number
attached to it is not. Calibration fixes the number without touching the
ranking.

Method: temperature scaling. A single scalar T divides the log-probabilities
before the softmax:

    p_i = softmax(log p_i / T)

T > 1 softens an over-confident distribution. Two properties make it the
right choice here rather than per-class isotonic regression:

  * it is monotone in the logits, so the differential's ORDER is unchanged
    -- the top-3 list the user sees cannot be reshuffled by calibration;
  * it is one parameter, so it cannot overfit the calibration split, and it
    is trivial to explain in a viva.

T is fitted on out-of-fold predictions over the training set (so no row is
used to calibrate a model that saw it), and Expected Calibration Error is
then reported on the held-out test set, which is touched only for scoring.

ONE TEMPERATURE IS NOT ENOUGH, and measuring showed why. On the test
distribution the model is already well calibrated (ECE 0.023, global T
~1.0). But test rows carry 6-7 symptoms, while a real user volunteers two
or three. Truncating each test case to its first k symptoms exposes the
real problem:

    k=2   confidence 0.591  accuracy 0.399   gap +0.193
    k=3   confidence 0.689  accuracy 0.563   gap +0.125
    k=5   confidence 0.829  accuracy 0.774   gap +0.055

The miscalibration is a function of how much evidence the model has, so a
single global scalar cannot correct it -- it would have to soften confident
8-symptom cases to fix sparse 2-symptom ones. A temperature is therefore
fitted PER EVIDENCE BUCKET, T(k), and the serving code picks the bucket from
the number of symptoms actually known. Still monotone, still one parameter
per bucket, still ranking-preserving.

Run:  backend/venv/Scripts/python.exe ml/calibrate.py
"""

import json
import os

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "model")
OUT = os.path.join(MODEL_DIR, "calibration.json")

RANDOM_STATE = 42
N_FOLDS = 5
N_BINS = 10


def load_csv(filename):
    import csv
    path = os.path.join(DATA_DIR, filename)
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols = [c for c in rows[0] if c != "prognosis"]
    X = np.array([[int(r[c]) for c in cols] for r in rows], dtype=float)
    y = np.array([r["prognosis"] for r in rows])
    return X, y, cols


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y_idx, T):
    p = softmax(logits / T)
    return float(-np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12).mean())


def fit_temperature(logits, y_idx, lo=0.5, hi=10.0, iters=60):
    """Golden-section search on a 1-D convex-ish objective."""
    gr = (np.sqrt(5.0) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if nll(logits, y_idx, c) < nll(logits, y_idx, d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return round((a + b) / 2, 4)


def reliability(probs, y_idx, n_bins=N_BINS):
    """Bin by predicted confidence; return ECE plus the diagram's points."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_idx).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"bin_lower": round(lo, 2), "bin_upper": round(hi, 2),
                         "count": 0, "confidence": None, "accuracy": None})
            continue
        c = float(conf[mask].mean())
        a = float(correct[mask].mean())
        ece += (n / len(conf)) * abs(a - c)
        bins.append({"bin_lower": round(lo, 2), "bin_upper": round(hi, 2),
                     "count": n, "confidence": round(c, 4), "accuracy": round(a, 4)})
    return round(float(ece), 4), bins


# Evidence buckets. A user with two symptoms and a user with nine are in
# genuinely different regimes; these are the boundaries the measurements
# above showed matter.
BUCKETS = [(1, 2, "1-2"), (3, 3, "3"), (4, 5, "4-5"), (6, 99, "6+")]


def truncate(X, k, seed=0):
    """Keep at most k of each row's positive symptoms -- a partial report."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X)
    for i in range(len(X)):
        on = np.flatnonzero(X[i])
        keep = on if len(on) <= k else rng.choice(on, k, replace=False)
        out[i, keep] = 1
    return out


def bucket_of(n: int) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= n <= hi:
            return label
    return BUCKETS[-1][2]


def main():
    metrics = json.load(open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8"))
    model_name = metrics.get("model", "BernoulliNB")
    factory = MultinomialNB if model_name == "MultinomialNB" else BernoulliNB
    alpha = metrics.get("alpha", 1.0)

    X_tr, y_tr, _ = load_csv("training.csv")
    X_te, y_te, _ = load_csv("testing.csv")

    classes = np.array(json.load(open(os.path.join(MODEL_DIR, "label_classes.json"),
                                      encoding="utf-8")))
    index = {c: i for i, c in enumerate(classes)}
    ytr_idx = np.array([index[v] for v in y_tr])
    yte_idx = np.array([index[v] for v in y_te])

    # --- out-of-fold log-probabilities on the training set ----------------
    oof = np.zeros((len(y_tr), len(classes)))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fit_idx, hold_idx in skf.split(X_tr, y_tr):
        m = factory(alpha=alpha)
        m.fit(X_tr[fit_idx], y_tr[fit_idx])
        lp = m.predict_log_proba(X_tr[hold_idx])
        # Align this fold's class order to the global one.
        cols = [index[c] for c in m.classes_]
        oof[np.ix_(hold_idx, cols)] = lp

    T = fit_temperature(oof, ytr_idx)

    # --- score on the held-out test set -----------------------------------
    model = joblib.load(os.path.join(MODEL_DIR, "nb_model.joblib"))
    logits = model.predict_log_proba(X_te)
    cols = [index[c] for c in model.classes_]
    aligned = np.full((len(y_te), len(classes)), -1e9)
    aligned[:, cols] = logits

    p_before = softmax(aligned)
    p_after = softmax(aligned / T)

    ece_before, bins_before = reliability(p_before, yte_idx)
    ece_after, bins_after = reliability(p_after, yte_idx)

    acc_before = float((p_before.argmax(1) == yte_idx).mean())
    acc_after = float((p_after.argmax(1) == yte_idx).mean())

    # --- evidence-conditional temperatures --------------------------------
    # Build a calibration set that looks like real usage: every training row
    # truncated to a plausible number of reported symptoms. Temperatures are
    # fitted here (training rows only) and scored on truncated TEST rows.
    per_bucket = {}
    fit_model = factory(alpha=alpha)
    fit_model.fit(X_tr, y_tr)
    fit_cols = [index[c] for c in fit_model.classes_]

    for lo, hi, label in BUCKETS:
        ks = range(lo, min(hi, 9) + 1)

        def stack(X, y_idx, seed_base):
            Ls, Ys = [], []
            for k in ks:
                Xk = truncate(X, k, seed=seed_base + k)
                lg = np.full((len(X), len(classes)), -1e9)
                lg[:, fit_cols] = fit_model.predict_log_proba(Xk)
                Ls.append(lg)
                Ys.append(y_idx)
            return np.vstack(Ls), np.concatenate(Ys)

        Lfit, Yfit = stack(X_tr, ytr_idx, 100)
        Tb = fit_temperature(Lfit, Yfit)

        Lte, Yte = stack(X_te, yte_idx, 900)
        p0 = softmax(Lte)
        p1 = softmax(Lte / Tb)
        e0, _ = reliability(p0, Yte)
        e1, b1 = reliability(p1, Yte)
        per_bucket[label] = {
            "symptoms_known": label,
            "temperature": Tb,
            "n_eval": int(len(Yte)),
            "ece_before": e0,
            "ece_after": e1,
            "mean_confidence_before": round(float(p0.max(1).mean()), 4),
            "mean_confidence_after": round(float(p1.max(1).mean()), 4),
            "accuracy": round(float((p1.argmax(1) == Yte).mean()), 4),
            "reliability_after": b1,
        }

    payload = {
        "method": "temperature scaling, conditioned on evidence count",
        "model": model_name,
        "buckets": [list(b) for b in BUCKETS],
        "per_bucket": per_bucket,
        "global_temperature": T,
        "temperature": T,
        "fitted_on": f"out-of-fold predictions over {len(y_tr)} training rows ({N_FOLDS}-fold)",
        "evaluated_on": f"{len(y_te)} held-out test rows",
        "n_bins": N_BINS,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_reduction": round(ece_before - ece_after, 4),
        "mean_confidence_before": round(float(p_before.max(1).mean()), 4),
        "mean_confidence_after": round(float(p_after.max(1).mean()), 4),
        "accuracy_before": round(acc_before, 4),
        "accuracy_after": round(acc_after, 4),
        "accuracy_unchanged": abs(acc_before - acc_after) < 1e-9,
        "reliability_before": bins_before,
        "reliability_after": bins_after,
        "note": (
            "Temperature scaling is monotone in the logits, so the ranked differential "
            "is identical before and after; only the probabilities attached to it change."
        ),
    }
    json.dump(payload, open(OUT, "w", encoding="utf-8"), indent=2)

    print("=" * 62)
    print("Probability calibration - temperature scaling")
    print("=" * 62)
    print(f"  model               : {model_name}")
    print(f"  temperature T       : {T}   ({'softening' if T > 1 else 'sharpening'})")
    print(f"  mean confidence     : {payload['mean_confidence_before']:.3f} -> {payload['mean_confidence_after']:.3f}")
    print(f"  ECE (test)          : {ece_before:.4f} -> {ece_after:.4f}   ({ece_before - ece_after:+.4f})")
    print(f"  accuracy (test)     : {acc_before:.4f} -> {acc_after:.4f}   (ranking preserved: {payload['accuracy_unchanged']})")
    print()
    print("  Evidence-conditional temperatures (the ones actually served):")
    print(f"    {'symptoms':>9} {'T':>7} {'conf before':>12} {'conf after':>11} {'ECE before':>11} {'ECE after':>10} {'acc':>7}")
    for label, b in per_bucket.items():
        print(f"    {label:>9} {b['temperature']:>7.3f} {b['mean_confidence_before']:>12.3f} "
              f"{b['mean_confidence_after']:>11.3f} {b['ece_before']:>11.4f} {b['ece_after']:>10.4f} "
              f"{b['accuracy']:>7.3f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
