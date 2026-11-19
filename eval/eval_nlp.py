"""
Evaluate the NLP components against a hand-annotated gold standard.

A classifier accuracy number says nothing about whether the *text
understanding* works. These are the metrics that do:

* **Symptom extraction** -- micro and macro precision / recall / F1 over
  symptom sets, plus negation-aware scoring: an extraction that happily
  returns a symptom the user explicitly denied is counted as a false
  positive, because that is exactly the failure that would mislead a user.
* **Assertion (NegEx/ConText)** -- accuracy and per-class breakdown over the
  six derived labels.
* **Intent** -- accuracy, macro F1 and the confusion pairs.
* **Temporal** -- exact-value accuracy on the normalized duration in days,
  and bucket accuracy on the questionnaire option it maps to.
* **Trend / distress** -- accuracy on the lexicon layers.

Results land in `eval/results/nlp_eval.json`, which the API serves.

Note on leakage: the intent gold set was written to avoid reusing training
phrasings, and this script asserts that none of its items appears verbatim
in `intent.TRAINING_EXAMPLES`. Items handled by the pre-model regex rules
are reported separately so the statistical model's own accuracy is visible.
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))

from app.nlp import intent as intent_module  # noqa: E402
from app.nlp import negation, sentiment, temporal  # noqa: E402
from app.nlp.ner import recognize  # noqa: E402
from app.nlp.tokenizer import tokenize_sentences  # noqa: E402

GOLD_PATH = os.path.join(HERE, "gold", "nlp_gold.json")
RESULTS_DIR = os.path.join(HERE, "results")


def load_gold():
    with open(GOLD_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------


def eval_extraction(cases):
    micro_tp = micro_fp = micro_fn = 0
    per_case = []
    macro_scores = []
    negation_violations = 0
    exact_matches = 0

    for case in cases:
        result = recognize(case["text"], use_semantic=False)
        predicted = set(result.present_symptoms)
        expected = set(case["expected"])
        denied = set(case.get("negated", []))

        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)

        # A symptom the user denied that still came through is the worst
        # possible error here, so count it explicitly as well.
        leaked = predicted & denied
        negation_violations += len(leaked)

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        if tp == fp == fn == 0:
            # Correctly extracting nothing (family history, a hypothetical)
            # is a success, not an undefined score.
            precision = recall = f1 = 1.0
        else:
            precision, recall, f1 = prf(tp, fp, fn)
        macro_scores.append((precision, recall, f1))
        if predicted == expected:
            exact_matches += 1

        per_case.append(
            {
                "text": case["text"],
                "expected": sorted(expected),
                "predicted": sorted(predicted),
                "missed": sorted(expected - predicted),
                "spurious": sorted(predicted - expected),
                "negation_leaks": sorted(leaked),
                "f1": round(f1, 3),
            }
        )

    micro = prf(micro_tp, micro_fp, micro_fn)
    n = len(macro_scores) or 1
    macro = (
        sum(s[0] for s in macro_scores) / n,
        sum(s[1] for s in macro_scores) / n,
        sum(s[2] for s in macro_scores) / n,
    )

    return {
        "n_cases": len(cases),
        "micro_precision": round(micro[0], 4),
        "micro_recall": round(micro[1], 4),
        "micro_f1": round(micro[2], 4),
        "macro_precision": round(macro[0], 4),
        "macro_recall": round(macro[1], 4),
        "macro_f1": round(macro[2], 4),
        "exact_set_match": round(exact_matches / len(cases), 4) if cases else 0.0,
        "true_positives": micro_tp,
        "false_positives": micro_fp,
        "false_negatives": micro_fn,
        "negation_violations": negation_violations,
        "cases": per_case,
    }


def eval_assertion(cases):
    correct = 0
    per_label = defaultdict(lambda: {"correct": 0, "total": 0})
    confusions = Counter()
    failures = []

    for case in cases:
        sentences = tokenize_sentences(case["text"].lower())
        if not sentences:
            continue
        sentence = sentences[0]
        target = case["target"].lower()
        start = case["text"].lower().find(target)
        if start < 0:
            failures.append({"text": case["text"], "error": "target not found"})
            continue
        assertion = negation.classify(sentence, start, start + len(target))
        got = assertion.label
        expected = case["expected"]

        per_label[expected]["total"] += 1
        if got == expected:
            correct += 1
            per_label[expected]["correct"] += 1
        else:
            confusions[f"{expected} -> {got}"] += 1
            failures.append(
                {
                    "text": case["text"],
                    "target": case["target"],
                    "expected": expected,
                    "got": got,
                    "cues": [c.text for c in assertion.cues],
                }
            )

    return {
        "n_cases": len(cases),
        "accuracy": round(correct / len(cases), 4) if cases else 0.0,
        "per_label": {
            label: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
                "support": v["total"],
            }
            for label, v in sorted(per_label.items())
        },
        "confusions": dict(confusions),
        "failures": failures,
    }


def eval_intent(cases):
    train_texts = {t.lower().strip() for t, _ in intent_module.TRAINING_EXAMPLES}
    leaked = [c["text"] for c in cases if c["text"].lower().strip() in train_texts]

    correct = 0
    rule_decided = 0
    rule_correct = 0
    model_total = 0
    model_correct = 0
    per_label = defaultdict(lambda: {"correct": 0, "total": 0})
    confusions = Counter()
    failures = []

    for case in cases:
        result = intent_module.classify(case["text"])
        got = result["intent"]
        expected = case["expected"]
        per_label[expected]["total"] += 1
        if result["source"] == "rule":
            rule_decided += 1
        else:
            model_total += 1

        if got == expected:
            correct += 1
            per_label[expected]["correct"] += 1
            if result["source"] == "rule":
                rule_correct += 1
            else:
                model_correct += 1
        else:
            confusions[f"{expected} -> {got}"] += 1
            failures.append(
                {
                    "text": case["text"],
                    "expected": expected,
                    "got": got,
                    "confidence": round(result["confidence"], 3),
                    "source": result["source"],
                }
            )

    # Macro F1 across labels.
    labels = sorted({c["expected"] for c in cases})
    predictions = [intent_module.classify(c["text"])["intent"] for c in cases]
    truths = [c["expected"] for c in cases]
    f1s = []
    for label in labels:
        tp = sum(1 for p, t in zip(predictions, truths) if p == label and t == label)
        fp = sum(1 for p, t in zip(predictions, truths) if p == label and t != label)
        fn = sum(1 for p, t in zip(predictions, truths) if p != label and t == label)
        f1s.append(prf(tp, fp, fn)[2])

    return {
        "n_cases": len(cases),
        "leaked_from_training": leaked,
        "accuracy": round(correct / len(cases), 4) if cases else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "rule_decided": rule_decided,
        "rule_accuracy": round(rule_correct / rule_decided, 4) if rule_decided else None,
        "model_decided": model_total,
        "model_accuracy": round(model_correct / model_total, 4) if model_total else None,
        "per_label": {
            label: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
                "support": v["total"],
            }
            for label, v in sorted(per_label.items())
        },
        "confusions": dict(confusions),
        "failures": failures,
    }


def eval_temporal(cases):
    value_correct = 0
    bucket_correct = 0
    failures = []

    for case in cases:
        expressions = temporal.extract(case["text"].lower())
        primary = temporal.primary_duration(expressions)
        got_days = primary.value if primary else None
        got_bucket = primary.normalized if primary else None
        expected_days = case["expected_days"]
        expected_bucket = case["expected_bucket"]

        value_ok = (
            (got_days is None and expected_days is None)
            or (
                got_days is not None
                and expected_days is not None
                and abs(got_days - expected_days) < 0.02
            )
        )
        bucket_ok = got_bucket == expected_bucket

        value_correct += value_ok
        bucket_correct += bucket_ok
        if not (value_ok and bucket_ok):
            failures.append(
                {
                    "text": case["text"],
                    "expected_days": expected_days,
                    "got_days": got_days,
                    "expected_bucket": expected_bucket,
                    "got_bucket": got_bucket,
                }
            )

    return {
        "n_cases": len(cases),
        "value_accuracy": round(value_correct / len(cases), 4) if cases else 0.0,
        "bucket_accuracy": round(bucket_correct / len(cases), 4) if cases else 0.0,
        "failures": failures,
    }


def eval_trend(cases):
    correct = 0
    failures = []
    for case in cases:
        expressions = temporal.extract(case["text"].lower())
        found = temporal.first_of(expressions, "trend")
        got = found.normalized if found else None
        if got == case["expected"]:
            correct += 1
        else:
            failures.append(
                {"text": case["text"], "expected": case["expected"], "got": got}
            )
    return {
        "n_cases": len(cases),
        "accuracy": round(correct / len(cases), 4) if cases else 0.0,
        "failures": failures,
    }


def eval_distress(cases):
    correct = 0
    failures = []
    for case in cases:
        got = sentiment.score(case["text"]).level
        if got == case["expected_level"]:
            correct += 1
        else:
            failures.append(
                {"text": case["text"], "expected": case["expected_level"], "got": got}
            )
    return {
        "n_cases": len(cases),
        "accuracy": round(correct / len(cases), 4) if cases else 0.0,
        "failures": failures,
    }


# ---------------------------------------------------------------------------


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gold = load_gold()

    print("=" * 78)
    print("NLP component evaluation against the gold standard")
    print("=" * 78)

    start = time.perf_counter()
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gold_version": gold.get("_version"),
        "symptom_extraction": eval_extraction(gold["symptom_extraction"]),
        "assertion": eval_assertion(gold["assertion"]),
        "intent": eval_intent(gold["intent"]),
        "temporal": eval_temporal(gold["temporal"]),
        "trend": eval_trend(gold["trend"]),
        "distress": eval_distress(gold["distress"]),
    }
    report["elapsed_s"] = round(time.perf_counter() - start, 2)

    extraction = report["symptom_extraction"]
    print(f"\nSymptom extraction  (n={extraction['n_cases']})")
    print(f"   micro P/R/F1     : {extraction['micro_precision']:.3f} / "
          f"{extraction['micro_recall']:.3f} / {extraction['micro_f1']:.3f}")
    print(f"   macro P/R/F1     : {extraction['macro_precision']:.3f} / "
          f"{extraction['macro_recall']:.3f} / {extraction['macro_f1']:.3f}")
    print(f"   exact set match  : {extraction['exact_set_match']:.3f}")
    print(f"   TP/FP/FN         : {extraction['true_positives']} / "
          f"{extraction['false_positives']} / {extraction['false_negatives']}")
    print(f"   negation leaks   : {extraction['negation_violations']}  "
          f"(denied symptoms that still reached the model)")

    assertion = report["assertion"]
    print(f"\nAssertion / ConText (n={assertion['n_cases']})")
    print(f"   accuracy         : {assertion['accuracy']:.3f}")
    for label, values in assertion["per_label"].items():
        print(f"      {label:<14} {values['accuracy']:.3f}  (n={values['support']})")

    intent_report = report["intent"]
    print(f"\nIntent classification (n={intent_report['n_cases']})")
    print(f"   accuracy         : {intent_report['accuracy']:.3f}")
    print(f"   macro F1         : {intent_report['macro_f1']:.3f}")
    print(f"   rule-decided     : {intent_report['rule_decided']} "
          f"(acc {intent_report['rule_accuracy']})")
    print(f"   model-decided    : {intent_report['model_decided']} "
          f"(acc {intent_report['model_accuracy']})")
    print(f"   leaked from train: {intent_report['leaked_from_training'] or 'none'}")

    temporal_report = report["temporal"]
    print(f"\nTemporal normalization (n={temporal_report['n_cases']})")
    print(f"   value accuracy   : {temporal_report['value_accuracy']:.3f}")
    print(f"   bucket accuracy  : {temporal_report['bucket_accuracy']:.3f}")

    print(f"\nTrend detection      : {report['trend']['accuracy']:.3f} "
          f"(n={report['trend']['n_cases']})")
    print(f"Distress detection   : {report['distress']['accuracy']:.3f} "
          f"(n={report['distress']['n_cases']})")

    # Surface every failure, so the report can discuss them honestly.
    sections = [
        ("assertion", report["assertion"]["failures"]),
        ("intent", report["intent"]["failures"]),
        ("temporal", report["temporal"]["failures"]),
        ("trend", report["trend"]["failures"]),
        ("distress", report["distress"]["failures"]),
    ]
    any_failures = any(items for _, items in sections)
    extraction_failures = [c for c in extraction["cases"] if c["f1"] < 1.0]
    if any_failures or extraction_failures:
        print("\n" + "-" * 78)
        print("Failures")
        print("-" * 78)
        for case in extraction_failures:
            print(f"   [extraction] {case['text']}")
            print(f"       missed={case['missed']} spurious={case['spurious']}")
        for name, items in sections:
            for item in items:
                print(f"   [{name}] {item}")

    out_path = os.path.join(RESULTS_DIR, "nlp_eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {out_path}  ({report['elapsed_s']}s)")


if __name__ == "__main__":
    main()
