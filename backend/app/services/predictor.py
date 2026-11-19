import json
import os
import logging
import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "model")

# Both Naive Bayes variants are trained by ml/train_model.py. Bernoulli is the
# default because it models symptom *absence*, which the NegEx layer supplies
# and which measured ~3 points higher macro F1 (see eval/compare_models.py).
_VARIANT_FILES = {
    "bernoulli": "nb_bernoulli.joblib",
    "multinomial": "nb_multinomial.joblib",
}

_model = None
_feature_order = None
_label_classes = None
_disease_symptom_matrix = None
_calibration = None

# Below this calibrated probability the system should not name a single
# condition. Chosen from the measurements in ml/calibrate.py: at two known
# symptoms the calibrated confidence averages ~0.36 and top-1 accuracy is
# ~0.35, so anything under 0.45 is genuinely a coin-toss between candidates.
ABSTAIN_BELOW = 0.45
# Fewer than this many symptoms is not enough evidence to name one condition,
# whatever the number says.
MIN_SYMPTOMS_TO_NAME = 2


def load_calibration():
    """Evidence-conditional temperatures written by ml/calibrate.py."""
    global _calibration
    path = os.path.join(MODEL_DIR, "calibration.json")
    if not os.path.exists(path):
        logger.info("No calibration.json; serving raw model probabilities.")
        _calibration = None
        return
    try:
        with open(path, encoding="utf-8") as fh:
            _calibration = json.load(fh)
        temps = {k: v["temperature"] for k, v in _calibration.get("per_bucket", {}).items()}
        logger.info("Loaded calibration (per-bucket temperatures: %s)", temps)
    except Exception as exc:
        logger.warning("Could not read calibration.json: %s", exc)
        _calibration = None


def _temperature_for(n_symptoms: int) -> float:
    """The temperature fitted for this much evidence."""
    if not _calibration:
        return 1.0
    for lo, hi, label in _calibration.get("buckets", []):
        if lo <= n_symptoms <= hi:
            return float(_calibration["per_bucket"][label]["temperature"])
    return float(_calibration.get("global_temperature", 1.0))


def get_calibration() -> dict:
    return _calibration or {}


def load_model():
    global _model, _feature_order, _label_classes, _disease_symptom_matrix

    # AROGYA_MODEL selects a variant; anything else falls back to the
    # default artifact that train_model.py wrote.
    variant = os.getenv("AROGYA_MODEL", "").strip().lower()
    model_path = os.path.join(MODEL_DIR, "nb_model.joblib")
    if variant in _VARIANT_FILES:
        candidate = os.path.join(MODEL_DIR, _VARIANT_FILES[variant])
        if os.path.exists(candidate):
            model_path = candidate
            logger.info("AROGYA_MODEL=%s -> loading %s", variant, _VARIANT_FILES[variant])
        else:
            logger.warning(
                "AROGYA_MODEL=%s requested but %s is missing; using default artifact",
                variant,
                _VARIANT_FILES[variant],
            )

    features_path = os.path.join(MODEL_DIR, "feature_order.json")
    labels_path = os.path.join(MODEL_DIR, "label_classes.json")
    matrix_path = os.path.join(MODEL_DIR, "disease_symptom_matrix.json")

    if not os.path.exists(model_path):
        logger.warning("Model artifacts not found. Predictor will return empty results.")
        return False

    _model = joblib.load(model_path)
    load_calibration()

    with open(features_path, "r", encoding="utf-8") as f:
        _feature_order = json.load(f)

    with open(labels_path, "r", encoding="utf-8") as f:
        _label_classes = json.load(f)

    if os.path.exists(matrix_path):
        with open(matrix_path, "r", encoding="utf-8") as f:
            _disease_symptom_matrix = json.load(f)

    logger.info(
        "Loaded %s with %d features, %d classes",
        type(_model).__name__,
        len(_feature_order),
        len(_label_classes),
    )
    return True


def get_metrics() -> dict:
    """The measured metrics saved alongside the deployed model."""
    path = os.path.join(MODEL_DIR, "metrics.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_all_metrics() -> dict:
    """Metrics for every trained variant, for the evaluation dashboard."""
    path = os.path.join(MODEL_DIR, "metrics_all.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def is_loaded() -> bool:
    return _model is not None


def get_feature_order() -> list[str]:
    return _feature_order or []


def get_label_classes() -> list[str]:
    return _label_classes or []


def get_model():
    """The fitted estimator itself, for the explainability layer."""
    return _model


def get_disease_symptom_matrix() -> dict:
    return _disease_symptom_matrix or {}


def predict(symptom_flags: dict[str, bool]) -> list[dict]:
    if _model is None or _feature_order is None:
        return []

    vector = np.zeros(len(_feature_order))
    for i, symptom in enumerate(_feature_order):
        if symptom in symptom_flags and symptom_flags[symptom]:
            vector[i] = 1

    vector_2d = vector.reshape(1, -1)

    # Apply the temperature fitted for this much evidence. Naive Bayes
    # multiplies correlated symptoms as if they were independent, which
    # inflates the winner most when few symptoms are known; see the measured
    # table in ml/calibrate.py. Temperature is monotone, so this changes the
    # numbers without reordering the differential.
    n_known = int(vector.sum())
    T = _temperature_for(n_known)
    if T and abs(T - 1.0) > 1e-6:
        logits = _model.predict_log_proba(vector_2d)[0] / T
        logits = logits - logits.max()
        exp = np.exp(logits)
        probabilities = exp / exp.sum()
    else:
        probabilities = _model.predict_proba(vector_2d)[0]

    top_indices = np.argsort(probabilities)[::-1][:5]
    results = []
    for idx in top_indices:
        results.append(
            {
                "disease": _label_classes[idx],
                "probability": round(float(probabilities[idx]), 4),
            }
        )

    return results


def should_abstain(predictions: list[dict], n_symptoms: int) -> dict:
    """
    Decide whether the evidence supports naming a single condition.

    Returns {abstain, reason, top_probability, n_symptoms}. Abstaining is a
    product decision as much as a statistical one: showing "GERD, 76%" from
    three common symptoms reads as a diagnosis, and the measured accuracy at
    that evidence level does not support it.
    """
    top = predictions[0]["probability"] if predictions else 0.0
    if n_symptoms < MIN_SYMPTOMS_TO_NAME:
        return {"abstain": True, "reason": "too_few_symptoms",
                "top_probability": top, "n_symptoms": n_symptoms}
    if top < ABSTAIN_BELOW:
        return {"abstain": True, "reason": "low_confidence",
                "top_probability": top, "n_symptoms": n_symptoms}
    return {"abstain": False, "reason": "", "top_probability": top,
            "n_symptoms": n_symptoms}


def get_discriminating_symptoms(
    candidates: list[str], mentioned_symptoms: list[str]
) -> list[str]:
    if not _disease_symptom_matrix:
        return []

    symptom_scores = {}
    for disease in candidates:
        if disease not in _disease_symptom_matrix:
            continue
        disease_symptoms = _disease_symptom_matrix[disease]
        for symptom in disease_symptoms:
            if symptom not in mentioned_symptoms:
                if symptom not in symptom_scores:
                    symptom_scores[symptom] = 0
                symptom_scores[symptom] += 1

    sorted_symptoms = sorted(symptom_scores.items(), key=lambda x: x[1], reverse=True)
    return [s[0] for s in sorted_symptoms[:5]]
