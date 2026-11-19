"""
Fit and save the intent classifier.

The pipeline trains this lazily if the artifact is missing, but that costs
~1.5s on the first request. Running this during setup moves the cost to
build time and makes the deployed model reproducible.

Reports held-out performance via stratified cross-validation, since the seed
corpus is small enough that a single split would be noise.
"""

import os
import sys

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.nlp import intent  # noqa: E402

FOLDS = 5
RANDOM_STATE = 42


def main():
    print("=" * 64)
    print("Arogya AI - intent classifier")
    print("=" * 64)

    examples = intent.TRAINING_EXAMPLES
    texts = np.array([t for t, _ in examples])
    labels = np.array([l for _, l in examples])

    print(f"\nExamples : {len(texts)}")
    print(f"Intents  : {len(set(labels.tolist()))}")
    counts = {label: int((labels == label).sum()) for label in sorted(set(labels.tolist()))}
    for label, count in counts.items():
        print(f"   {label:<20} {count}")

    smallest = min(counts.values())
    folds = min(FOLDS, smallest)
    if folds < 2:
        print("\nToo few examples in some class for cross-validation; skipping.")
    else:
        print(f"\nRunning {folds}-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        predicted = cross_val_predict(
            intent.build_pipeline(), texts, labels, cv=cv
        )
        accuracy = float((predicted == labels).mean())
        print(f"\nCross-validated accuracy: {accuracy:.4f}")
        print("\nPer-intent report (cross-validated):")
        print(classification_report(labels, predicted, zero_division=0))

    print("Fitting on the full corpus and saving...")
    intent.train_and_save()
    print(f"Saved {intent.MODEL_PATH}")

    print("\nSpot checks:")
    for probe in [
        "I have a bad headache and fever",
        "what should I eat now",
        "should I see a doctor",
        "no",
        "thanks a lot",
        "who won the cricket match",
    ]:
        result = intent.classify(probe)
        print(f"   {probe:<36} -> {result['intent']:<18} "
              f"{result['confidence']:.2f} ({result['source']})")


if __name__ == "__main__":
    main()
