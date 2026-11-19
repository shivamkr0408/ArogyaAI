"""
Train the production Naive Bayes classifiers.

Both variants are trained and saved:

* **Bernoulli NB** (deployed by default) models each symptom as present *or
  absent*. Absence is informative here -- "no fever" genuinely separates
  conditions -- and the NegEx/ConText layer in `backend/app/nlp/negation.py`
  produces exactly that signal, so the classifier can use it. It measured
  ~3 points higher macro F1 than Multinomial in `eval/compare_models.py`.
* **Multinomial NB** is kept for comparison and can be deployed instead by
  setting `AROGYA_MODEL=multinomial`.

Both decompose exactly, so the per-symptom explanation in the UI stays a
real attribution either way.

This also writes `ml/model/metrics.json`: accuracy, macro and weighted
precision/recall/F1, top-3 accuracy, cross-validated accuracy with its
standard deviation, the per-class report and the confusion matrix. The API
serves that file, so the numbers quoted in the UI and in the report are the
ones this script actually measured rather than hand-copied.

Top-3 accuracy is reported because the product shows a differential of
candidate conditions, not a single label -- "the right answer was in the
list we showed" is the metric that matches what the user sees.
"""

import csv
import json
import os
import time

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

os.makedirs(MODEL_DIR, exist_ok=True)

ALPHA = 1.0
CV_FOLDS = 5
RANDOM_STATE = 42

# (key, display name, factory, artifact filename)
VARIANTS = [
    ("bernoulli", "BernoulliNB", lambda: BernoulliNB(alpha=ALPHA), "nb_bernoulli.joblib"),
    ("multinomial", "MultinomialNB", lambda: MultinomialNB(alpha=ALPHA), "nb_multinomial.joblib"),
]

# Which variant `nb_model.joblib` (the file the backend loads by default)
# points at. Chosen on measured macro F1; see eval/compare_models.py.
DEFAULT_VARIANT = "bernoulli"


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    features, labels = [], []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        symptom_cols = [c for c in reader.fieldnames if c != "prognosis"]
        for row in reader:
            features.append([int(row[s]) for s in symptom_cols])
            labels.append(row["prognosis"])
    return np.array(features), np.array(labels), symptom_cols


def build_disease_symptom_matrix(X, y, symptom_cols, threshold=0.3):
    """Symptoms present in at least `threshold` of a disease's rows."""
    matrix = {}
    for disease in np.unique(y):
        mask = y == disease
        avg_presence = X[mask].mean(axis=0)
        matrix[disease] = [
            symptom_cols[i] for i, v in enumerate(avg_presence) if v >= threshold
        ]
    return matrix


def top_k_accuracy(model, X, y, k=3):
    """Fraction of rows whose true label is in the model's top k."""
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    hits = 0
    for row, truth in zip(probabilities, y):
        top_k = np.argsort(row)[::-1][:k]
        if truth in [classes[i] for i in top_k]:
            hits += 1
    return hits / len(y)


def fit_and_score(name, factory, symptom_cols, X_train, y_train, X_test, y_test, cv):
    """Fit one variant and return (model, metrics dict)."""
    print(f"\nFitting {name}(alpha={ALPHA})...")
    start = time.perf_counter()
    model = factory()
    model.fit(X_train, y_train)
    fit_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_ms = (time.perf_counter() - start) * 1000

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    top3 = top_k_accuracy(model, X_test, y_test, k=3)

    print(f"\nFit time         : {fit_ms:.1f} ms")
    print(f"Predict time     : {predict_ms:.1f} ms for {len(y_test)} rows")
    print(f"\nAccuracy         : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"Top-3 accuracy   : {top3:.4f}  ({top3*100:.1f}%)")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Weighted F1      : {weighted_f1:.4f}")
    print(f"Macro precision  : {macro_precision:.4f}")
    print(f"Macro recall     : {macro_recall:.4f}")

    print(f"Running {CV_FOLDS}-fold stratified cross-validation...")
    cv_scores = cross_val_score(
        factory(), X_train, y_train, cv=cv, scoring="accuracy"
    )
    print(f"CV accuracy      : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print("\nWeakest classes by F1:")
    per_class = [
        (label, values["f1-score"], int(values["support"]))
        for label, values in report.items()
        if isinstance(values, dict) and label not in ("accuracy", "macro avg", "weighted avg")
    ]
    for label, score, support in sorted(per_class, key=lambda kv: kv[1])[:5]:
        print(f"   {label:<34} F1={score:.3f}  (n={support})")

    labels_sorted = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    matrix = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    label_classes = model.classes_.tolist()

    metrics = {
        "model": name,
        "alpha": ALPHA,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": len(symptom_cols),
        "n_classes": len(label_classes),
        "accuracy": round(float(accuracy), 4),
        "top3_accuracy": round(float(top3), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "cv_folds": CV_FOLDS,
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_scores": [round(float(s), 4) for s in cv_scores],
        "fit_ms": round(fit_ms, 2),
        "predict_ms_total": round(predict_ms, 2),
        "predict_ms_per_row": round(predict_ms / max(1, len(y_test)), 4),
        "per_class": {
            label: {
                "precision": round(values["precision"], 4),
                "recall": round(values["recall"], 4),
                "f1": round(values["f1-score"], 4),
                "support": int(values["support"]),
            }
            for label, values in report.items()
            if isinstance(values, dict)
            and label not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": matrix.tolist(),
        },
    }
    return model, metrics


def main():
    print("=" * 64)
    print("Arogya AI - Naive Bayes training (both variants)")
    print("=" * 64)

    X_train, y_train, symptom_cols = load_csv("training.csv")
    X_test, y_test, _ = load_csv("testing.csv")

    print(f"\nTraining samples : {len(y_train)}")
    print(f"Testing samples  : {len(y_test)}")
    print(f"Features         : {len(symptom_cols)}")
    print(f"Classes          : {len(np.unique(y_train))}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    trained: dict[str, tuple] = {}
    for key, name, factory, filename in VARIANTS:
        print("\n" + "-" * 64)
        model, metrics = fit_and_score(
            name, factory, symptom_cols, X_train, y_train, X_test, y_test, cv
        )
        joblib.dump(model, os.path.join(MODEL_DIR, filename))
        trained[key] = (model, metrics, name, filename)
        print(f"Saved {filename}")

    # --- shared artifacts -------------------------------------------------
    default_model, default_metrics, default_name, default_file = trained[DEFAULT_VARIANT]
    label_classes = default_model.classes_.tolist()
    disease_symptom_matrix = build_disease_symptom_matrix(X_train, y_train, symptom_cols)

    # nb_model.joblib is what the backend loads unless AROGYA_MODEL overrides.
    joblib.dump(default_model, os.path.join(MODEL_DIR, "nb_model.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_order.json"), "w", encoding="utf-8") as fh:
        json.dump(symptom_cols, fh)
    with open(os.path.join(MODEL_DIR, "label_classes.json"), "w", encoding="utf-8") as fh:
        json.dump(label_classes, fh)
    with open(os.path.join(MODEL_DIR, "disease_symptom_matrix.json"), "w", encoding="utf-8") as fh:
        json.dump(disease_symptom_matrix, fh, indent=2)

    default_metrics["deployed"] = True
    default_metrics["variant_key"] = DEFAULT_VARIANT
    default_metrics["artifact"] = default_file
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(default_metrics, fh, indent=2)

    all_metrics = {
        "default_variant": DEFAULT_VARIANT,
        "variants": {
            key: {**metrics, "variant_key": key, "artifact": filename}
            for key, (_, metrics, _, filename) in trained.items()
        },
    }
    with open(os.path.join(MODEL_DIR, "metrics_all.json"), "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)

    print("\n" + "=" * 64)
    print("Variant comparison")
    print("=" * 64)
    print(f"{'Variant':<18}{'Acc':>8}{'Top-3':>9}{'MacroF1':>10}{'CV acc':>18}")
    for key, (_, metrics, name, _) in trained.items():
        marker = "  <- deployed" if key == DEFAULT_VARIANT else ""
        print(
            f"{name:<18}{metrics['accuracy']:>8.3f}{metrics['top3_accuracy']:>9.3f}"
            f"{metrics['macro_f1']:>10.3f}"
            f"{metrics['cv_accuracy_mean']:>12.3f}+/-{metrics['cv_accuracy_std']:.3f}"
            f"{marker}"
        )
    print(f"\nDeployed: {default_name} -> nb_model.joblib")
    print("Override with AROGYA_MODEL=multinomial (or =bernoulli)")

    # --- sanity check -----------------------------------------------------
    model = default_model
    print("\n" + "=" * 64)
    print(f"Sanity check on hand-built symptom sets ({default_name})")
    print("=" * 64)
    test_cases = [
        {"headache": True, "nausea": True, "blurred_vision": True},
        {"cough": True, "breathlessness": True, "chest_tightness": True, "wheezing": True},
        {"joint_pain": True, "joint_swelling": True, "joint_stiffness": True},
        {"itching": True, "skin_rash": True, "nodal_eruptions": True},
        {"high_fever": True, "chills": True, "sweating": True, "vomiting": True},
    ]
    for i, symptoms in enumerate(test_cases, 1):
        vector = np.zeros(len(symptom_cols))
        for j, col in enumerate(symptom_cols):
            if symptoms.get(col):
                vector[j] = 1
        probabilities = model.predict_proba(vector.reshape(1, -1))[0]
        print(f"\n{i}. {', '.join(sorted(symptoms))}")
        for rank, idx in enumerate(np.argsort(probabilities)[::-1][:3], 1):
            print(f"   {rank}. {label_classes[idx]:<32} {probabilities[idx]:.3f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
