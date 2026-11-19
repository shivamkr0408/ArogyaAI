"""
Synthetic symptom-disease dataset generator.

The first version of this script marked every core symptom of a disease as
present in 100% of that disease's rows. That makes the dataset trivially
separable -- every classifier scores ~100%, model comparison shows no
differences, and because P(symptom | disease) is identical for all core
symptoms, the Naive Bayes explanation cannot say which symptom mattered.

Real presentations are partial. So each symptom now has a *prevalence* for
its disease, assigned by tier:

    hallmark   0.90 - 0.98   nearly always present
    common     0.72 - 0.88
    frequent   0.52 - 0.72
    variable   0.32 - 0.52
    occasional 0.18 - 0.35

plus a low background rate for symptoms outside the disease's profile, so
rows carry realistic noise. Class sizes vary too, which gives the metrics
something to measure (macro vs weighted F1 stop being the same number).

Everything is seeded, so the dataset is reproducible.
"""

import csv
import json
import os
import random

SEED = 42
random.seed(SEED)

DISEASES = {
    "Fungal infection": ["itching", "skin_rash", "nodal_eruptions", "dischromic_patches", "skin_peeling"],
    "Allergy": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes", "skin_rash"],
    "GERD": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
    "Chronic cholestasis": ["itching", "pale_stools", "yellowing_of_eyes", "yellowing_of_skin", "vomiting", "loss_of_appetite", "abdominal_pain"],
    "Drug Reaction": ["itching", "skin_rash", "stomach_pain", "burning_micturition", "face_swelling"],
    "Peptic ulcer diseae": ["vomiting", "indigestion", "loss_of_appetite", "abdominal_pain", "passage_of_gases"],
    # Hallmark symptom first: prevalence is assigned by position, so the
    # order here encodes which symptom is most characteristic. Listing
    # muscle_pain first made it AIDS's hallmark, which let a 3-day febrile
    # illness with body aches be reported as AIDS at 85% confidence.
    "AIDS": ["weight_loss", "high_fever", "loss_of_appetite", "skin_rash", "joint_pain", "muscle_pain"],
    "Diabetes": ["increased_appetite", "polyuria", "weight_loss", "fatigue", "blurred_vision", "obesity"],
    "Gastroenteritis": ["vomiting", "loss_of_appetite", "nausea", "diarrhea", "abdominal_pain", "high_fever"],
    # "Bronchial Asthma" and "Asthma" are both in the source label set and
    # were near-duplicates, which capped both at ~0.56-0.70 f1 while they
    # stole each other's probability mass. They are kept as separate labels
    # but given the two presentations that genuinely differ: an acute
    # exacerbation that a reliever is not controlling, versus the chronic
    # episodic picture with night-time and trigger-linked symptoms.
    "Bronchial Asthma": ["reliever_not_helping", "breathlessness", "cough", "sweating", "fatigue", "high_fever", "chest_pain"],
    "Hypertension": ["headache", "chest_pain", "dizziness", "nausea", "fatigue"],
    "Migraine": ["acidity", "neck_pain", "vomiting", "headache", "blurred_vision", "irritability"],
    "Cervical spondylosis": ["back_pain", "weakness_in_limbs", "neck_pain", "headache", "dizziness", "fatigue"],
    "Paralysis (brain hemorrhage)": ["vomiting", "headache", "neck_pain", "weakness_of_body", "altered_sensorium", "dizziness"],
    "Jaundice": ["itching", "fatigue", "weight_loss", "high_fever", "vomiting", "yellowing_of_eyes"],
    "Malaria": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "diarrhea"],
    "Chicken pox": ["skin_rash", "moderate_fever", "malaise", "loss_of_appetite", "headache"],
    "Dengue": ["skin_rash", "high_fever", "joint_pain", "vomiting", "fatigue", "headache", "muscle_pain"],
    "Typhoid": ["high_fever", "headache", "malaise", "loss_of_appetite", "abdominal_pain", "constipation", "diarrhea"],
    # The hepatitis family used to be mutually indistinguishable: every
    # variant was drawn from the same pool of joint_pain / vomiting /
    # fatigue / nausea, and Hepatitis D's profile was a strict subset of the
    # others', so it scored f1 = 0.000 -- no classifier could ever separate
    # it. Each variant now carries the exposure or history that actually
    # distinguishes it in practice, placed first so it reads as the hallmark.
    "Hepatitis A": ["yellowing_of_eyes", "recent_contaminated_food", "joint_pain", "vomiting", "fatigue", "high_fever", "loss_of_appetite"],
    "Hepatitis B": ["unsafe_injection_exposure", "dark_urine", "fatigue", "vomiting", "nausea", "loss_of_appetite", "abdominal_pain", "high_fever"],
    "Hepatitis C": ["blood_transfusion_history", "fatigue", "dark_urine", "loss_of_appetite", "joint_pain", "abdominal_pain", "nausea"],
    # Hepatitis D only ever occurs alongside an existing hepatitis B
    # infection, so a known B diagnosis is its true discriminator.
    "Hepatitis D": ["known_hepatitis_b", "yellowing_of_eyes", "fatigue", "joint_pain", "vomiting", "abdominal_pain", "nausea"],
    "Hepatitis E": ["recent_contaminated_water", "yellowing_of_eyes", "joint_pain", "vomiting", "high_fever", "fatigue", "nausea"],
    "Alcoholic hepatitis": ["heavy_alcohol_use", "yellowing_of_eyes", "abdominal_pain", "vomiting", "loss_of_appetite", "nausea", "fever"],
    "Tuberculosis": ["cough_over_two_weeks", "night_sweats", "weight_loss", "cough", "loss_of_appetite", "high_fever", "fatigue", "chest_pain"],
    "Common Cold": ["continuous_sneezing", "chills", "fatigue", "cough", "runny_nose", "congestion", "headache", "sore_throat"],
    "Pneumonia": ["productive_sputum", "high_fever", "chills", "cough", "breathlessness", "chest_pain", "fatigue", "loss_of_appetite"],
    "Dimorphic hemmorhoids(piles)": ["constipation", "pain_during_bowel_movement", "bloody_stool", "anal_discomfort", "swelling"],
    "Heart attack": ["chest_pain", "breathlessness", "sweating", "nausea", "dizziness", "vomiting"],
    "Vertigo": ["spinning_movements", "vomiting", "headache", "nausea", "sweating", "dizziness"],
    "Urinary tract infection": ["burning_micturition", "foul_smell_of_urine", "continuous_feel_of_urine", "bladder_discomfort", "high_fever"],
    "Psoriasis": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
    "Impetigo": ["skin_rash", "blister", "red_sore_around_nose", "yellow_crust_sortation", "itching"],
    "Arthritis": ["joint_pain", "joint_swelling", "joint_stiffness", "loss_of_range_of_motion", "fatigue"],
    "Gout": ["joint_pain", "joint_swelling", "redness", "warmth", "tenderness", "fever"],
    "Asthma": ["wheezing", "night_time_symptoms", "symptoms_with_triggers", "cough", "breathlessness", "chest_tightness", "fatigue", "fast_heart_rate"],
    "COPD": ["smoking_history", "chronic_productive_cough", "breathlessness", "cough", "wheezing", "chest_tightness", "fatigue", "frequent_infections"],
    "Anxiety": ["anxiety", "palpitations", "sweating", "tremor", "insomnia", "restlessness", "fatigue"],
    "Depression": ["depression", "fatigue", "loss_of_appetite", "insomnia", "irritability", "weight_loss"],
    "Diabetic foot": ["numbness", "slow_healing", "skin_changes", "swelling", "pain", "foul_smelling_wound"],
}

SYMPTOMS = sorted({s for symptoms in DISEASES.values() for s in symptoms})

# Prevalence band by position in the disease's symptom list. Symptoms are
# listed roughly most- to least-characteristic, so position is a reasonable
# proxy for how reliably the symptom shows up.
PREVALENCE_TIERS = [
    (0.90, 0.98),   # hallmark
    (0.72, 0.88),   # common
    (0.52, 0.72),   # frequent
    (0.32, 0.52),   # variable
    (0.18, 0.35),   # occasional
]
TAIL_TIER = (0.12, 0.28)

# Symptoms outside the disease profile still turn up sometimes.
BACKGROUND_RANGE = (0.01, 0.05)

# ---------------------------------------------------------------------------
# Clinically informed base rates.
#
# Class sizes used to be random, which made P(disease) arbitrary -- and with
# Naive Bayes the class prior is a term in every decision. A randomly large
# AIDS class is why "fever and body aches for 3 days" could come back as AIDS
# at 85% confidence. Encoding roughly realistic frequencies for a primary-care
# setting makes the prior carry information instead of noise, so the model
# stops over-predicting rare, serious conditions.
#
# These are deliberately coarse ordinal tiers, not epidemiological estimates.
# ---------------------------------------------------------------------------

# The tiers stay ordinal, so P(disease) still carries base-rate information
# and the model still needs strong evidence before naming a rare condition.
# The floor was raised, though: at 20-32 rows a rare class had only 4-6 test
# rows, so its per-class F1 moved in steps of ~0.2 and was dominated by
# sampling noise rather than by what the model had learned.
TIER_SAMPLES = {
    "common": (150, 185),
    "moderate": (90, 120),
    "uncommon": (62, 82),
    "rare": (40, 55),
}

DISEASE_TIER = {
    # Things a walk-in clinic sees constantly.
    "Common Cold": "common",
    "Allergy": "common",
    "Gastroenteritis": "common",
    "GERD": "common",
    "Migraine": "common",
    "Urinary tract infection": "common",
    "Fungal infection": "common",
    # Frequent but not everyday.
    "Bronchial Asthma": "moderate",
    "Asthma": "moderate",
    "Hypertension": "moderate",
    "Diabetes": "moderate",
    "Arthritis": "moderate",
    "Cervical spondylosis": "moderate",
    "Anxiety": "moderate",
    "Depression": "moderate",
    "Vertigo": "moderate",
    "Dimorphic hemmorhoids(piles)": "moderate",
    "Peptic ulcer diseae": "moderate",
    "Drug Reaction": "moderate",
    "Typhoid": "moderate",
    "Dengue": "moderate",
    "Malaria": "moderate",
    "Chicken pox": "moderate",
    "Impetigo": "moderate",
    "Psoriasis": "moderate",
    "Jaundice": "moderate",
    # Less common presentations.
    "Pneumonia": "uncommon",
    "Gout": "uncommon",
    "COPD": "uncommon",
    "Hepatitis A": "uncommon",
    "Hepatitis E": "uncommon",
    "Tuberculosis": "uncommon",
    "Diabetic foot": "uncommon",
    "Chronic cholestasis": "uncommon",
    # Rare, serious, or requiring specialist diagnosis. A symptom checker
    # should need strong evidence before naming any of these.
    "AIDS": "rare",
    "Heart attack": "rare",
    "Paralysis (brain hemorrhage)": "rare",
    "Hepatitis B": "rare",
    "Hepatitis C": "rare",
    "Hepatitis D": "rare",
    "Alcoholic hepatitis": "rare",
}

DEFAULT_TIER = "moderate"

TEST_FRACTION = 0.2


def build_prevalence() -> dict[str, dict[str, float]]:
    """Assign a per-disease prevalence to every symptom. Deterministic."""
    rng = random.Random(SEED)
    table: dict[str, dict[str, float]] = {}
    for disease, core in DISEASES.items():
        profile: dict[str, float] = {}
        for position, symptom in enumerate(core):
            low, high = (
                PREVALENCE_TIERS[position]
                if position < len(PREVALENCE_TIERS)
                else TAIL_TIER
            )
            profile[symptom] = round(rng.uniform(low, high), 3)
        for symptom in SYMPTOMS:
            if symptom not in profile:
                profile[symptom] = round(rng.uniform(*BACKGROUND_RANGE), 3)
        table[disease] = profile
    return table


def generate_rows(prevalence: dict[str, dict[str, float]]):
    """Sample rows per disease by Bernoulli draws on each symptom."""
    rng = random.Random(SEED + 1)
    rows = []
    class_sizes = {}
    for disease, profile in prevalence.items():
        tier = DISEASE_TIER.get(disease, DEFAULT_TIER)
        low, high = TIER_SAMPLES[tier]
        n_samples = rng.randint(low, high)
        class_sizes[disease] = n_samples
        produced = 0
        attempts = 0
        while produced < n_samples and attempts < n_samples * 20:
            attempts += 1
            row = {s: int(rng.random() < profile[s]) for s in SYMPTOMS}
            # A row with almost nothing in it is not a usable presentation.
            if sum(row.values()) < 2:
                continue
            row["prognosis"] = disease
            rows.append(row)
            produced += 1
    rng.shuffle(rows)
    return rows, class_sizes


def stratified_split(rows, test_fraction=TEST_FRACTION):
    """Split per class, so every disease appears in both files."""
    rng = random.Random(SEED + 2)
    by_class: dict[str, list] = {}
    for row in rows:
        by_class.setdefault(row["prognosis"], []).append(row)

    train, test = [], []
    for disease, items in by_class.items():
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * test_fraction)))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_csv(filename, rows):
    path = os.path.join(os.path.dirname(__file__), "data", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SYMPTOMS + ["prognosis"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")
    return path


def main():
    print("=" * 64)
    print("Arogya AI - synthetic dataset generation (prevalence-weighted)")
    print("=" * 64)

    prevalence = build_prevalence()
    rows, class_sizes = generate_rows(prevalence)
    train, test = stratified_split(rows)

    write_csv("training.csv", train)
    write_csv("testing.csv", test)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    with open(os.path.join(data_dir, "symptom_prevalence.json"), "w", encoding="utf-8") as fh:
        json.dump(prevalence, fh, indent=2, sort_keys=True)

    # Average symptoms per row, and how often the hallmark symptom is absent
    # -- the two numbers that show the dataset is no longer deterministic.
    avg_symptoms = sum(sum(r[s] for s in SYMPTOMS) for r in rows) / len(rows)
    hallmark_absent = 0
    for row in rows:
        hallmark = DISEASES[row["prognosis"]][0]
        if not row[hallmark]:
            hallmark_absent += 1

    stats = {
        "seed": SEED,
        "total_rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "features": len(SYMPTOMS),
        "classes": len(DISEASES),
        "class_sizes": class_sizes,
        "class_tiers": {d: DISEASE_TIER.get(d, DEFAULT_TIER) for d in DISEASES},
        "tier_sample_ranges": TIER_SAMPLES,
        "smallest_class": min(class_sizes.values()),
        "largest_class": max(class_sizes.values()),
        "avg_symptoms_per_row": round(avg_symptoms, 2),
        "rows_missing_hallmark_symptom": hallmark_absent,
        "pct_missing_hallmark_symptom": round(100.0 * hallmark_absent / len(rows), 2),
    }
    with open(os.path.join(data_dir, "dataset_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print()
    print(f"Classes              : {stats['classes']}")
    print(f"Features             : {stats['features']}")
    print(f"Rows                 : {stats['total_rows']} "
          f"(train {stats['train_rows']} / test {stats['test_rows']})")
    print(f"Class sizes          : {stats['smallest_class']}-{stats['largest_class']} per disease")
    print(f"Avg symptoms per row : {stats['avg_symptoms_per_row']}")
    print(f"Hallmark absent in   : {stats['pct_missing_hallmark_symptom']}% of rows "
          f"(partial presentations)")
    print()
    print("Wrote symptom_prevalence.json and dataset_stats.json")


if __name__ == "__main__":
    main()
