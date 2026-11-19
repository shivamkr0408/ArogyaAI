"""
Tests for the Naive Bayes explanation layer.

The important property is *exactness*: the per-symptom weights this module
reports must reconstruct the estimator's own joint log-likelihood. If that
ever stops holding, the explanation panel is lying to the user, and no
amount of plausible-looking output would reveal it.
"""

import numpy as np
import pytest

from app.services import explain, predictor


@pytest.fixture(scope="module", autouse=True)
def loaded_model():
    if not predictor.load_model():
        pytest.skip("model artifacts missing - run python ml/train_model.py")
    return predictor.get_model()


FLAGS = {
    "high_fever": True,
    "chills": True,
    "sweating": True,
    "vomiting": True,
    "headache": True,
}


def test_explanation_has_one_entry_per_reported_symptom():
    result = explain.explain(FLAGS)
    assert result is not None
    reported = {s for s, on in FLAGS.items() if on}
    assert {c["symptom"] for c in result["contributions"]} == reported


def test_decomposition_reconstructs_the_model_score_exactly():
    """
    Rebuild the estimator's joint log-likelihood from the reported weights.

    Bernoulli: prior + sum(log(1-theta)) + sum over present of log-odds.
    Multinomial: prior + sum over present of log P(symptom | class).
    """
    model = predictor.get_model()
    features = predictor.get_feature_order()
    classes = predictor.get_label_classes()

    result = explain.explain(FLAGS)
    index = classes.index(result["condition"])

    vector = np.array([1.0 if FLAGS.get(f) else 0.0 for f in features])
    sklearn_score = float(model._joint_log_likelihood(vector.reshape(1, -1))[0][index])

    reported = sum(c["log_likelihood"] for c in result["contributions"])
    rebuilt = float(model.class_log_prior_[index]) + reported

    if type(model).__name__ == "BernoulliNB":
        log_theta = np.asarray(model.feature_log_prob_[index])
        neg = np.log(np.clip(1.0 - np.exp(log_theta), 1e-12, None))
        rebuilt += float(neg.sum())

    # Reported weights are rounded to 4 decimals for display.
    assert rebuilt == pytest.approx(sklearn_score, abs=1e-3)


def test_discrimination_is_relative_to_the_runner_up():
    result = explain.explain(FLAGS)
    assert result["compared_with"]
    assert result["compared_with"] != result["condition"]
    for contribution in result["contributions"]:
        assert contribution["discrimination"] is not None


def test_support_shares_sum_to_one():
    result = explain.explain(FLAGS)
    total = sum(c["support_share"] for c in result["contributions"])
    assert total == pytest.approx(1.0, abs=0.01)


def test_no_symptoms_yields_no_explanation():
    assert explain.explain({}) is None


def test_denied_symptoms_are_reported_as_evidence_for_bernoulli():
    model = predictor.get_model()
    if type(model).__name__ != "BernoulliNB":
        pytest.skip("absence evidence is a Bernoulli-only property")
    result = explain.explain(FLAGS, negated_symptoms=["cough", "diarrhea"])
    labels = {d["symptom"] for d in result["denial_evidence"]}
    assert labels == {"cough", "diarrhea"}


def test_summary_names_the_driving_symptoms():
    result = explain.explain(FLAGS)
    summary = explain.summarize(result)
    assert result["condition"] in summary
    assert any(c["label"] in summary for c in result["top_discriminating"])


def test_expected_but_not_reported_excludes_reported_symptoms():
    result = explain.explain(FLAGS)
    missing = {m["symptom"] for m in result["expected_but_not_reported"]}
    assert not (missing & {s for s, on in FLAGS.items() if on})
