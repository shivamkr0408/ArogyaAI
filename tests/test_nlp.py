"""
Unit tests for the NLP pipeline.

These cover the behaviours that are easy to regress silently and expensive to
get wrong -- negation scope, code-mixed normalization, offset mapping -- and
the Porter stemmer against its published reference vocabulary.

The aggregate quality numbers live in `eval/eval_nlp.py`, which scores the
whole gold standard; these are the fast checks that run on every change.
"""

import pytest

from app.nlp import lexicon, sentiment, temporal
from app.nlp.negation import classify
from app.nlp.ner import recognize
from app.nlp.normalizer import normalize
from app.nlp.pipeline import analyze
from app.nlp.tokenizer import lemma, stem, tokenize, tokenize_sentences


# ---------------------------------------------------------------- tokenizer

# Pairs from Porter's published voc.txt / output.txt.
PORTER_REFERENCE = [
    ("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"),
    ("caress", "caress"), ("cats", "cat"), ("feed", "feed"),
    ("agreed", "agre"), ("plastered", "plaster"), ("bled", "bled"),
    ("motoring", "motor"), ("sing", "sing"), ("conflated", "conflat"),
    ("troubled", "troubl"), ("sized", "size"), ("hopping", "hop"),
    ("tanned", "tan"), ("failing", "fail"), ("filing", "file"),
    ("happy", "happi"), ("sky", "sky"), ("relational", "relat"),
    ("conditional", "condit"), ("rational", "ration"),
    ("digitizer", "digit"), ("vietnamization", "vietnam"),
    ("predication", "predic"), ("operator", "oper"), ("feudalism", "feudal"),
    ("decisiveness", "decis"), ("hopefulness", "hope"),
    ("callousness", "callous"), ("triplicate", "triplic"),
    ("formative", "form"), ("formalize", "formal"), ("hopeful", "hope"),
    ("goodness", "good"), ("revival", "reviv"), ("allowance", "allow"),
    ("inference", "infer"), ("airliner", "airlin"), ("gyroscopic", "gyroscop"),
    ("adjustable", "adjust"), ("defensible", "defens"), ("irritant", "irrit"),
    ("replacement", "replac"), ("adjustment", "adjust"),
    ("dependent", "depend"), ("adoption", "adopt"), ("communism", "commun"),
    ("activate", "activ"), ("effective", "effect"), ("bowdlerize", "bowdler"),
    ("probate", "probat"), ("rate", "rate"), ("cease", "ceas"),
    ("controll", "control"), ("roll", "roll"),
]


@pytest.mark.parametrize("word,expected", PORTER_REFERENCE)
def test_porter_matches_reference_vocabulary(word, expected):
    assert stem(word) == expected


@pytest.mark.parametrize(
    "forms,expected",
    [
        (["vomiting", "vomited", "vomits", "vomit"], "vomit"),
        (["coughing", "coughed", "coughs", "cough"], "cough"),
        (["aching", "ached", "aches", "ache"], "ache"),
        (["itching", "itchy", "itches"], "itch"),
        (["swelling", "swollen", "swells"], "swell"),
    ],
)
def test_clinical_forms_collapse_to_one_lemma(forms, expected):
    assert {lemma(f) for f in forms} == {expected}


def test_token_offsets_round_trip():
    text = "I have no fever. But my head hurts since 3 days!"
    for token in tokenize(text):
        assert text[token.start : token.end] == token.text


def test_sentences_are_split_and_tokens_attributed():
    sentences = tokenize_sentences("No fever. My head hurts.")
    assert len(sentences) == 2
    assert all(s.tokens for s in sentences)


# --------------------------------------------------------------- normalizer


@pytest.mark.parametrize(
    "raw,expected_fragment",
    [
        ("mujhe pet dard hai", "stomach pain"),
        ("tez bukhar aur khansi", "high fever"),
        ("saans lene me takleef", "breathlessness"),
        ("I have a feaver", "fever"),
        ("coughhh", "cough"),
        ("I don't have fever", "do not"),
        ("बुखार", "fever"),
    ],
)
def test_normalization_rewrites(raw, expected_fragment):
    assert expected_fragment in normalize(raw).text


@pytest.mark.parametrize(
    "word",
    ["cold", "hand", "head", "back", "water", "doctor", "sleep", "food", "last",
     "mother", "week", "rahi", "hain", "nahi", "aur"],
)
def test_protected_words_are_never_corrected(word):
    """Spell correction must not mangle ordinary English or Hindi grammar."""
    assert normalize(word).text == word


def test_offsets_map_back_through_a_code_mixed_rewrite():
    """The whole point of the alignment table: spans point at what was typed."""
    result = normalize("mujhe pet dard aur tez bukhar hai")
    index = result.text.find("stomach pain")
    start, end = result.to_original_span(index, index + len("stomach pain"))
    assert result.original[start:end] == "pet dard"


# ----------------------------------------------------------------- negation


@pytest.mark.parametrize(
    "text,target,expected",
    [
        ("i have a fever", "fever", "affirmed"),
        ("i have no fever", "fever", "negated"),
        ("i do not have any fever", "fever", "negated"),
        ("denies chest pain", "chest pain", "negated"),
        ("without any vomiting", "vomiting", "negated"),
        ("fever is absent", "fever", "negated"),
        ("my father has diabetes", "diabetes", "family"),
        ("family history of asthma", "asthma", "family"),
        ("if i get a fever should i worry", "fever", "hypothetical"),
        ("i am worried about dengue", "dengue", "hypothetical"),
        ("i had dengue last year", "dengue", "historical"),
        ("i used to get migraines", "migraines", "historical"),
        ("i think i have a fever", "fever", "uncertain"),
        ("no change in my cough", "cough", "affirmed"),
        ("my father never had asthma", "asthma", "negated"),
    ],
)
def test_assertion_labels(text, target, expected):
    sentence = tokenize_sentences(text)[0]
    index = text.find(target)
    assert classify(sentence, index, index + len(target)).label == expected


def test_termination_stops_negation_scope():
    """"no cough but I do have a fever" must not negate the fever."""
    text = "no cough but i do have a fever"
    sentence = tokenize_sentences(text)[0]
    cough = text.find("cough")
    fever = text.find("fever")
    assert classify(sentence, cough, cough + 5).label == "negated"
    assert classify(sentence, fever, fever + 5).label == "affirmed"


def test_coordination_does_not_stop_negation_scope():
    """"and" coordinates same-polarity items, so both must stay negated."""
    text = "denies vomiting and nausea"
    sentence = tokenize_sentences(text)[0]
    for target in ("vomiting", "nausea"):
        index = text.find(target)
        assert classify(sentence, index, index + len(target)).label == "negated"


def test_duration_is_not_read_as_a_past_episode():
    text = "i have had a fever since last week"
    sentence = tokenize_sentences(text)[0]
    index = text.find("fever")
    assert classify(sentence, index, index + 5).label == "affirmed"


def test_context_axes_are_independent():
    text = "my father never had asthma"
    sentence = tokenize_sentences(text)[0]
    index = text.find("asthma")
    assertion = classify(sentence, index, index + len("asthma"))
    assert assertion.negation == "negated"
    assert assertion.experiencer == "other"
    assert not assertion.is_clinically_present


# ----------------------------------------------------------------- temporal


@pytest.mark.parametrize(
    "text,days,bucket",
    [
        ("fever since 3 days", 3.0, "1-3 days"),
        ("cough for the past two weeks", 14.0, "More than 2 weeks"),
        ("headache since yesterday", 1.0, "1-3 days"),
        ("do din se bukhar", 2.0, "1-3 days"),
        ("teen din se khansi", 3.0, "1-3 days"),
        ("pain for about 6 hours", 0.25, "Less than a day"),
        ("started a couple of days ago", 2.0, "1-3 days"),
        ("fever since 1 week", 7.0, "About a week"),
    ],
)
def test_duration_normalization(text, days, bucket):
    primary = temporal.primary_duration(temporal.extract(text))
    assert primary is not None
    assert primary.value == pytest.approx(days, abs=0.02)
    assert primary.normalized == bucket


def test_bare_past_reference_is_not_a_duration():
    assert temporal.primary_duration(temporal.extract("i had dengue last year")) is None


@pytest.mark.parametrize(
    "text,expected",
    [("it is getting worse", "worsening"), ("about the same", "unchanged"),
     ("feeling better now", "improving")],
)
def test_trend_detection(text, expected):
    found = temporal.first_of(temporal.extract(text), "trend")
    assert found is not None and found.normalized == expected


# ---------------------------------------------------------------------- NER


def test_recognizer_excludes_negated_symptoms_from_model_input():
    result = recognize("I don't have any fever but my chest hurts", use_semantic=False)
    assert "chest_pain" in result.present_symptoms
    assert "fever" not in result.present_symptoms
    assert "fever" in result.negated_symptoms


def test_code_mixed_entities_keep_original_spans():
    result = recognize("mujhe tez bukhar aur khansi hai", use_semantic=False)
    assert set(result.present_symptoms) == {"high_fever", "cough"}
    surfaces = {e.original_text for e in result.symptoms}
    assert "tez bukhar" in surfaces


def test_family_history_is_not_attributed_to_the_patient():
    result = recognize("my father has diabetes, I feel fine", use_semantic=False)
    assert result.present_symptoms == []
    conditions = result.mentioned_conditions
    assert conditions and conditions[0]["attribution"] == "family"


def test_hypothetical_symptoms_do_not_reach_the_model():
    result = recognize("if I get a fever should I worry", use_semantic=False)
    assert result.present_symptoms == []


def test_lone_distress_word_is_not_a_clinical_symptom():
    """"I'm really worried" is commentary, not an anxiety complaint."""
    result = recognize("I'm really worried, I have a high fever", use_semantic=False)
    assert "anxiety" not in result.present_symptoms
    assert "high_fever" in result.present_symptoms


def test_described_anxiety_is_still_recognized():
    result = recognize("I am feeling anxious all the time", use_semantic=False)
    assert "anxiety" in result.present_symptoms


def test_no_appetite_is_a_symptom_not_a_negation():
    result = recognize("no appetite and losing weight", use_semantic=False)
    assert "loss_of_appetite" in result.present_symptoms


# ---------------------------------------------------------------- sentiment


@pytest.mark.parametrize(
    "text,level",
    [
        ("I have a mild cough for 2 days", "calm"),
        ("I am not worried, just checking", "calm"),
        ("I am very anxious about these symptoms", "elevated"),
        ("I am really scared, is this cancer?", "high"),
        ("the pain is unbearable and I cant cope", "high"),
    ],
)
def test_distress_levels(text, level):
    assert sentiment.score(text).level == level


# ----------------------------------------------------------------- pipeline


def test_pipeline_reports_every_stage():
    analysis = analyze("I have a high fever since 3 days", include_tokens=True)
    assert analysis.present_symptoms == ["high_fever"]
    assert analysis.duration_bucket == "1-3 days"
    assert analysis.intent["intent"]
    assert analysis.timings and analysis.total_ms > 0
    assert analysis.tokens


def test_pipeline_is_fast_enough_for_a_request():
    """The hot path must stay well under a perceptible delay."""
    analyze("warm up", include_tokens=False)
    analysis = analyze(
        "tez bukhar aur badan dard since 3 days, no cough", include_tokens=False
    )
    assert analysis.total_ms < 400


# ------------------------------------------------------------------ lexicon


def test_every_alias_maps_to_a_real_model_feature():
    """A typo'd canonical name would silently never fire."""
    features = set(lexicon.canonical_symptoms())
    unknown = {v for v in lexicon.phrase_index().values() if v not in features}
    assert not unknown, f"phrase index targets unknown features: {sorted(unknown)}"


def test_contraction_variants_are_registered():
    """Normalization expands contractions before lookup, so both must exist."""
    index = lexicon.phrase_index()
    assert "cant sleep" in index and "can not sleep" in index
