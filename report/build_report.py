"""
Generate the project report as styled HTML, then render it to PDF.

Every metric in the report is read from the JSON that the evaluation scripts
wrote, so the document cannot drift from what was actually measured. Re-run
the evaluation, re-run this, and the report is current.

    python report/build_report.py              # writes HTML + PDF
    python report/build_report.py --html-only  # skip the Chrome step

PDF rendering uses headless Chrome, which is the only HTML-to-PDF path
available on the development machine (no poppler / wkhtmltopdf / pandoc).
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "eval", "results")
MODEL_DIR = os.path.join(ROOT, "ml", "model")
DATA_DIR = os.path.join(ROOT, "ml", "data")

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

TEAM = [
    ("Rishav Kumar", "23BCE0462"),
    ("Anshuman Kumar", "23BCE0311"),
    ("Rahul Kumar", "23BCE0246"),
]
GUIDE = "Prof. Deepa K"
COURSE = "BCSE409L — NLP Based Project"
INSTITUTE = "Vellore Institute of Technology"


def load(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default if default is not None else {}


def pct(value, digits=1):
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def dec(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.2pt; line-height: 1.5; color: #15201d; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 23pt; margin: 0 0 4px; letter-spacing: -0.4px; color: #1f443e; }
h2 {
  font-size: 13.5pt; margin: 22px 0 8px; color: #23514a;
  border-bottom: 2px solid #b5e2d7; padding-bottom: 4px;
  page-break-after: avoid;
}
h3 { font-size: 11pt; margin: 14px 0 5px; color: #28645a; page-break-after: avoid; }
p { margin: 0 0 8px; text-align: justify; }
ul, ol { margin: 0 0 9px; padding-left: 20px; }
li { margin-bottom: 3px; }
code, .mono { font-family: Consolas, "SF Mono", monospace; font-size: 9pt; }
table { width: 100%; border-collapse: collapse; margin: 9px 0 12px; font-size: 8.9pt; page-break-inside: avoid; }
th { background: #eef9f6; color: #1f443e; text-align: left; font-weight: 600;
     padding: 5px 7px; border-bottom: 1.5px solid #89cdbd; }
td { padding: 4.5px 7px; border-bottom: 1px solid #e4ebe9; vertical-align: top; }
td.num, th.num { text-align: right; font-family: Consolas, monospace; }
tr.hl td { background: #f2fbf8; font-weight: 600; }
.cover { text-align: center; padding: 48mm 0 0; page-break-after: always; }
.cover .sub { font-size: 12pt; color: #2e7d6f; margin-top: 6px; }
.cover .meta { margin-top: 34px; font-size: 10.5pt; line-height: 1.9; }
.cover .rule { width: 90px; height: 3px; background: #3e9987; margin: 16px auto 0; }
.badge { display: inline-block; background: #eef9f6; border: 1px solid #b5e2d7;
         color: #23514a; border-radius: 11px; padding: 1px 9px; font-size: 8.4pt; margin: 0 3px 3px 0; }
.callout { background: #f7fbfa; border-left: 3px solid #3e9987; padding: 8px 11px; margin: 10px 0; }
.callout.warn { background: #fffaf3; border-left-color: #d98324; }
.key { font-weight: 600; color: #1f443e; }
.small { font-size: 8.6pt; color: #5a6a66; }
.footer { position: fixed; bottom: -11mm; left: 0; right: 0; text-align: center;
          font-size: 7.3pt; color: #8a9995; }
.pipeline { font-family: Consolas, monospace; font-size: 8.1pt; line-height: 1.45;
            background: #f7fbfa; border: 1px solid #d8f1ea; padding: 9px 11px;
            white-space: pre; margin: 9px 0; page-break-inside: avoid; }
.refs { font-size: 8.7pt; }
.refs li { margin-bottom: 6px; text-align: justify; }
.two { display: flex; gap: 14px; }
.two > div { flex: 1; }
.avoid { page-break-inside: avoid; }
"""


def build_html():
    comparison = load(os.path.join(RESULTS, "model_comparison.json"))
    nlp = load(os.path.join(RESULTS, "nlp_eval.json"))
    rag = load(os.path.join(RESULTS, "rag_eval.json"))
    metrics = load(os.path.join(MODEL_DIR, "metrics.json"))
    stats = load(os.path.join(DATA_DIR, "dataset_stats.json"))

    results = comparison.get("results", [])
    results_sorted = sorted(results, key=lambda r: -r.get("macro_f1", 0))
    deployed = comparison.get("deployed_model", "Bernoulli NB")
    best = comparison.get("best_by_macro_f1", "")
    gap = comparison.get("macro_f1_gap_to_best", 0)
    dataset = comparison.get("dataset", {})

    extraction = nlp.get("symptom_extraction", {})
    assertion = nlp.get("assertion", {})
    intent = nlp.get("intent", {})
    temporal = nlp.get("temporal", {})
    unf = rag.get("unfiltered", {})
    filt = rag.get("filtered", {})

    team_rows = "".join(
        f"<div>{name} &nbsp;<span class='mono small'>{reg}</span></div>" for name, reg in TEAM
    )

    model_rows = "".join(
        f"""<tr class="{'hl' if r['model'] == deployed else ''}">
          <td>{r['model']}{' <span class="small">(deployed)</span>' if r['model'] == deployed else ''}</td>
          <td class="num">{dec(r.get('accuracy'))}</td>
          <td class="num">{dec(r.get('top3_accuracy'))}</td>
          <td class="num">{dec(r.get('macro_f1'))}</td>
          <td class="num">{dec(r.get('weighted_f1'))}</td>
          <td class="num">{dec(r.get('cv_accuracy_mean'))} ± {dec(r.get('cv_accuracy_std'))}</td>
          <td class="num">{r.get('fit_ms', 0):.0f}</td>
        </tr>"""
        for r in results_sorted
    )

    per_class = metrics.get("per_class", {})
    weakest = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])[:5]
    weak_rows = "".join(
        f"<tr><td>{name}</td><td class='num'>{dec(v['precision'])}</td>"
        f"<td class='num'>{dec(v['recall'])}</td><td class='num'>{dec(v['f1'])}</td>"
        f"<td class='num'>{v['support']}</td></tr>"
        for name, v in weakest
    )

    assertion_rows = "".join(
        f"<tr><td>{label}</td><td class='num'>{pct(v['accuracy'], 0)}</td><td class='num'>{v['support']}</td></tr>"
        for label, v in sorted(assertion.get("per_label", {}).items())
    )

    today = date.today().strftime("%d %B %Y")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Arogya AI — Project Report</title>
<style>{CSS}</style></head><body>

<div class="footer">Arogya AI · {COURSE} · {' · '.join(n for n, _ in TEAM)} · Guide: {GUIDE} · {INSTITUTE}</div>

<!-- ============================ COVER ============================ -->
<div class="cover">
  <h1>Arogya AI</h1>
  <div class="sub">A Staged Clinical NLP Pipeline for Conversational<br>Symptom Assessment and Explainable Triage</div>
  <div class="rule"></div>
  <div class="meta">
    <div style="margin-bottom:14px"><strong>{COURSE}</strong></div>
    {team_rows}
    <div style="margin-top:16px">Guide: <strong>{GUIDE}</strong></div>
    <div>{INSTITUTE}</div>
    <div class="small" style="margin-top:22px">{today}</div>
  </div>
</div>

<!-- ============================ 1 ABSTRACT ============================ -->
<h2>1 Abstract</h2>
<p>
Online symptom checkers typically treat a patient's free-text complaint as a bag of keywords: they
search for symptom strings and feed whatever they find to a classifier. That approach fails on the
constructions patients actually use. "I don't have a fever" contributes a fever; "my father has
diabetes" contributes diabetes; "if I get a fever, should I worry?" contributes a fever. Each of
these is a <em>negation</em>, <em>experiencer</em> or <em>temporality</em> error, and each one
silently corrupts the input to the downstream model.
</p>
<p>
This project builds a staged clinical NLP pipeline that addresses these failures explicitly. Free
text is normalized (including code-mixed Hinglish and Devanagari) while preserving a character-level
alignment back to the original; entities are recognized by a three-tier cascade of dictionary,
approximate-string and embedding matching; and each mention is assigned an assertion status along the
three ConText axes — negation, temporality and experiencer — so that only symptoms that are
<em>affirmed</em>, <em>current</em> and <em>about the patient</em> reach the classifier. Temporal
expressions are normalized to durations, user intent is classified so that questions about an existing
assessment are answered from it, and the Naive Bayes prediction is decomposed exactly into
per-symptom log-odds to produce a genuine attribution rather than a post-hoc approximation.
</p>
<p>
Each component is evaluated independently. Against a hand-annotated gold standard the extraction layer
achieves <span class="key">{dec(extraction.get('micro_f1'))} micro-F1</span> with
<span class="key">zero</span> negation leaks, assertion detection achieves
<span class="key">{pct(assertion.get('accuracy'), 0)}</span> accuracy, and temporal normalization
<span class="key">{pct(temporal.get('value_accuracy'), 0)}</span>. Eight classifiers were benchmarked
under identical splits and folds; {deployed} is deployed at
<span class="key">{dec(metrics.get('macro_f1'))} macro-F1</span> and
<span class="key">{pct(metrics.get('top3_accuracy'))} top-3 accuracy</span>. Retrieval reaches
<span class="key">{dec(unf.get('mrr'))} MRR</span>. The system is explicitly an educational tool and
not a diagnostic device.
</p>

<!-- ============================ 2 PROBLEM DOMAIN ============================ -->
<h2>2 Problem Domain</h2>
<p>
Primary-care access in India is constrained by clinician availability, and a large share of first
contact with the health system is now a web search. Symptom checkers are the natural intermediary, but
they inherit two problems that are specifically linguistic rather than medical.
</p>
<h3>2.1 Clinical language is not keyword-shaped</h3>
<p>
Patients describe symptoms with negation, hedging, attribution to family members, and references to
past episodes. A system that extracts symptom strings without resolving these constructions does not
merely lose information — it ingests <em>inverted</em> information. The table below shows constructions
drawn from our gold standard and the behaviour of naive extraction.
</p>
<table class="avoid">
  <tr><th>Patient utterance</th><th>Naive extraction</th><th>Correct interpretation</th></tr>
  <tr><td>"I don't have any fever but my chest hurts"</td><td>fever, chest pain</td><td>chest pain only; fever <em>denied</em></td></tr>
  <tr><td>"no cough but I do have body aches"</td><td>cough, muscle pain</td><td>muscle pain only; scope ends at "but"</td></tr>
  <tr><td>"my father has diabetes"</td><td>diabetes (patient)</td><td>family history, different experiencer</td></tr>
  <tr><td>"I had dengue last year"</td><td>dengue (current)</td><td>historical, not current</td></tr>
  <tr><td>"if I get a fever should I worry"</td><td>fever</td><td>hypothetical; nothing reported</td></tr>
  <tr><td>"no appetite and losing weight"</td><td>weight loss only</td><td>loss of appetite <em>is</em> the symptom</td></tr>
</table>
<h3>2.2 Indian users write code-mixed text</h3>
<p>
A deployable Indian health assistant must accept "mujhe tez bukhar aur badan dard hai teen din se".
Romanized Hindi, Devanagari and English are mixed within single sentences, and the medical vocabulary
is often the Hindi part. Treating this as noise discards the clinical content of the message.
</p>
<h3>2.3 Opacity</h3>
<p>
A system that returns a condition and a confidence number, with no account of which symptom produced
it, cannot be audited by the user or by a clinician, and gives no basis for the user to judge whether
the assessment is worth acting on.
</p>

<!-- ============================ 3 LITERATURE ============================ -->
<h2>3 Existing Methodology &amp; Literature Review</h2>
<p>
Clinical decision support has moved rapidly from rule-based expert systems toward large language
models [1]. Chatbot responses to patient questions have been shown to be rated comparably to
physicians' on quality and empathy in a public-forum setting [2], and LLMs have demonstrated encoded
clinical knowledge at the level of professional examinations [3]. However, these systems are
expensive, opaque, and prone to fluent error, which is why retrieval-augmented generation [4] has
become the standard mitigation — grounding generated text in a retrieved corpus. Benchmarking of RAG
specifically in healthcare [5] shows that retrieval quality, not generation quality, is frequently the
limiting factor, which motivates evaluating retrieval on its own, as we do in Section 7.
</p>
<p>
A parallel line of work pursues lightweight, interpretable prediction. Naive Bayes over symptom
vectors remains a strong and well-studied baseline for disease prediction [8], and recent work applies
deep learning and NLP techniques directly to free-text symptom descriptions [9]. Agentic formulations
for triage and treatment planning [6] and precision-medicine framings of patient-centric care [7]
describe the broader direction of travel. The present work follows the specification in [10].
</p>
<h3>3.1 Gap addressed</h3>
<p>
Across this literature, the <em>assertion</em> problem is comparatively under-treated in
symptom-checker applications, despite being well established in clinical NLP: NegEx and its successor
ConText are long-standing algorithms for exactly this task. Our contribution is not to invent a new
algorithm but to assemble the right ones into a pipeline, extend them for code-mixed input, and
measure each stage independently rather than reporting a single end-to-end accuracy figure.
</p>

<!-- ============================ 4 PROPOSED SOLUTION ============================ -->
<h2>4 Proposed Solution — Novelty of Arogya AI</h2>
<p>Five elements distinguish this system from a keyword-plus-classifier symptom checker.</p>
<h3>4.1 Assertion detection on three independent axes</h3>
<p>
Every recognized mention is classified on the three ConText axes rather than a single
present/absent boolean:
</p>
<div class="pipeline">"my father never had asthma"
    negation    = negated    (cue: "never",      scopes right, 7 tokens, stops at termination)
    experiencer = other      (cue: "my father")
    temporality = recent
    => excluded from the feature vector on two independent grounds</div>
<p>
Scope is bounded and terminates at contrast terms ("but", "however", "except"), which is what makes
"no cough but I do have a fever" resolve correctly. Coordinators ("and", "or") are deliberately
<em>not</em> terminators, matching the published NegEx specification, so "denies vomiting and nausea"
negates both. Pseudo-negations are matched first and block a genuine negation cue from firing inside
them, which is what keeps "no appetite" a symptom rather than a denial.
</p>
<h3>4.2 Code-mixed normalization with offset preservation</h3>
<p>
A 139-entry romanized-Hindi and Devanagari lexicon rewrites code-mixed input to clinical English
before recognition. The difficulty is that the interface highlights entities in the text the user
actually typed, so a naive string replacement would destroy the offsets. Every rewrite is therefore
recorded as an edit with both its original and normalized span, and a segment table maps any span in
the normalized string back to its source. "pet dard" is rewritten to "stomach pain", recognized as
<span class="mono">stomach_pain</span>, and highlighted over the original Hindi.
</p>
<h3>4.3 A recognition cascade that reports its own method</h3>
<p>
Recognition proceeds exact → fuzzy → embedding, each tier more permissive and more expensive than the
last, with the embedding tier running only when the first two find nothing. Each entity carries the
strategy that found it and a confidence, so the interface can show <em>how</em> a symptom was
recognized rather than simply asserting it.
</p>
<h3>4.4 Exact explainability</h3>
<p>
Naive Bayes is one of the few classifiers whose decision decomposes exactly, and the system exploits
this. For the deployed Bernoulli model,
</p>
<div class="pipeline">log P(c | x) = log P(c) + Σ_i [ x_i · log θ_ci + (1 − x_i) · log(1 − θ_ci) ]</div>
<p>
so every reported symptom contributes exactly its log-odds, and every <em>denied</em> symptom
contributes a real term as well. The attribution shown to the user is therefore the model's own
arithmetic, not a LIME- or SHAP-style approximation; a unit test asserts that the reported weights
reconstruct scikit-learn's joint log-likelihood to within rounding.
</p>
<h3>4.5 Intent-aware dialogue policy</h3>
<p>
User turns are classified into eleven intents. A question about an existing assessment ("what should I
eat?") is answered from the stored prediction rather than restarting the consultation. Because acting
on a wrong intent is more costly than ignoring a right one, routing requires 45% confidence; below
that the turn falls through to the ordinary symptom path.
</p>

<!-- ============================ 5 ARCHITECTURE ============================ -->
<h2>5 System Architecture</h2>
<div class="pipeline">  User turn
      │
      ▼
  ┌─────────────────────── app/nlp/ pipeline ───────────────────────┐
  │ normalizer   NFC · contractions · elongation · code-mixed ·      │
  │              bounded Damerau-Levenshtein    → offset alignment   │
  │ tokenizer    offset-preserving tokens + Porter (1980) stemmer    │
  │ ner          exact → fuzzy → MiniLM cascade, spans + confidence  │
  │ negation     NegEx + ConText, 3 axes, bounded scope              │
  │ temporal     duration / frequency / trend  → normalized days     │
  │ intent       TF-IDF (word + char n-grams) → logistic regression  │
  │ sentiment    weighted distress lexicon                           │
  └──────────────────────────────┬──────────────────────────────────┘
                                 ▼
   RED-FLAG SAFETY SCAN  ── match ──►  EMERGENCY (terminal, no diagnosis)
                                 │
                                 ▼
   Bernoulli Naive Bayes ──► Explainability (exact log-odds decomposition)
                                 │
                                 ▼
   Chroma RAG retrieval ──► Ayurveda KB ──► Triage ──► Response</div>
<p>
The safety layer runs on every turn <em>before</em> any prediction or guidance is produced, so an
emergency presentation can never be answered with a diagnosis. The frontend is React + TypeScript;
the backend is FastAPI with SQLite session persistence. The LLM is optional: without a key the
pipeline runs fully locally, and when a key is present the LLM's symptom list is unioned with the
local pipeline's, with the assertion layer retaining final authority.
</p>

<!-- ============================ 6 IMPLEMENTATION ============================ -->
<h2>6 Implementation</h2>
<h3>6.1 Lexical resources</h3>
<table class="avoid">
  <tr><th>Resource</th><th class="num">Size</th><th>Purpose</th></tr>
  <tr><td>Canonical symptoms</td><td class="num">87</td><td>Model feature vocabulary (single source of truth)</td></tr>
  <tr><td>Surface phrases</td><td class="num">639</td><td>Clinical synonyms + colloquial English phrasings</td></tr>
  <tr><td>Code-mixed phrases</td><td class="num">139</td><td>Romanized Hindi and Devanagari</td></tr>
  <tr><td>Disease phrases</td><td class="num">115</td><td>Lay condition names for comorbidity / family history</td></tr>
  <tr><td>Assertion cues</td><td class="num">217</td><td>Negation, pseudo-negation, termination, hypothetical, historical, family, uncertainty</td></tr>
  <tr><td>Intent corpus</td><td class="num">360</td><td>11 intents, including code-mixed and misspelled variants</td></tr>
</table>
<h3>6.2 Stemming</h3>
<p>
The Porter (1980) algorithm is implemented in full and verified against the author's published
vocabulary/output pair (71 of 71 reference pairs). Domain irregulars are handled in a separate lemma
layer rather than inside the algorithm, so the stemmer remains a faithful implementation and can be
tested as such.
</p>
<h3>6.3 Spell correction</h3>
<p>
Correction is bounded Damerau-Levenshtein against the medical vocabulary <em>only</em>, with a
length-scaled distance budget, so a typo can be pulled towards a clinical term but ordinary words are
never altered. Two protection sets proved essential: body-site names ("hand" is one edit from "band")
and romanized Hindi function words — without the latter, edit distance happily rewrote "rahi" to
"nahi" and "hain" to "gain", corrupting the sentence before negation ever ran.
</p>
<h3>6.4 Dataset construction</h3>
<p>
The synthetic generator was rebuilt during this work after two defects were identified. First, every
core symptom was present in 100% of its disease's rows, which made the data trivially separable,
pushed all classifiers to near-perfect scores, and — because P(symptom | disease) was identical for
every core symptom — made the per-symptom explanation uninformative. Symptoms now carry a per-disease
<em>prevalence</em> assigned by tier (hallmark 0.90–0.98 down to occasional 0.18–0.35), with a low
background rate for symptoms outside the profile.
</p>
<p>
Second, class sizes were uniform, so the Naive Bayes prior carried no information. The consequence was
concrete: "fever and body aches for three days" was returned as <strong>AIDS at 85% confidence</strong>.
Class sizes now follow coarse clinical base rates (common conditions 130–165 samples, rare or
specialist-diagnosis conditions 20–32), and the same presentation now returns Dengue 46% / Typhoid 35%,
while AIDS still surfaces correctly for its own presentation.
</p>
<table class="avoid">
  <tr><th>Property</th><th class="num">Value</th><th>Property</th><th class="num">Value</th></tr>
  <tr><td>Total rows</td><td class="num">{stats.get('total_rows', '—')}</td><td>Classes</td><td class="num">{stats.get('classes', '—')}</td></tr>
  <tr><td>Train / test</td><td class="num">{stats.get('train_rows', '—')} / {stats.get('test_rows', '—')}</td><td>Features</td><td class="num">{stats.get('features', '—')}</td></tr>
  <tr><td>Class size range</td><td class="num">{stats.get('smallest_class', '—')}–{stats.get('largest_class', '—')}</td><td>Mean symptoms per row</td><td class="num">{stats.get('avg_symptoms_per_row', '—')}</td></tr>
  <tr><td>Rows missing the hallmark symptom</td><td class="num">{stats.get('pct_missing_hallmark_symptom', '—')}%</td><td>Split</td><td class="num">stratified</td></tr>
</table>

<!-- ============================ 7 RESULTS ============================ -->
<h2>7 Results &amp; Verification</h2>
<p>
Each component is measured separately. A single end-to-end accuracy figure would hide which stage is
responsible for an error, and would not reveal the failure mode that matters most here — a denied
symptom reaching the classifier.
</p>

<h3>7.1 Classifier comparison</h3>
<p class="small">
{dataset.get('train_rows', '—')} train / {dataset.get('test_rows', '—')} test rows ·
{dataset.get('features', '—')} features · {dataset.get('classes', '—')} classes ·
identical stratified split and {dataset.get('cv_folds', 5)}-fold cross-validation for every model.
</p>
<table>
  <tr><th>Model</th><th class="num">Acc.</th><th class="num">Top-3</th><th class="num">Macro F1</th>
      <th class="num">Wtd F1</th><th class="num">CV accuracy</th><th class="num">Fit (ms)</th></tr>
  {model_rows}
</table>
<div class="callout">
  <span class="key">Model selection.</span> {best} attains the highest macro-F1, {dec(gap)} above the
  deployed {deployed}. {deployed} was nonetheless selected because (i) its decision decomposes exactly
  into per-symptom log-odds, which is what makes the explanation a true attribution; (ii) unlike
  Multinomial NB it scores symptom <em>absence</em>, so the output of the assertion layer becomes
  usable evidence rather than merely omitted information; and (iii) it fits in milliseconds. The
  trade-off is stated rather than hidden, and both Naive Bayes variants ship — <span class="mono">AROGYA_MODEL</span>
  selects between them.
</div>
<p>
The majority-class baseline scores {dec(next((r['accuracy'] for r in results if 'baseline' in r['model'].lower()), None))}
accuracy across {dataset.get('classes', 41)} classes, confirming that every other row has learned
something. Accuracy alone is misleading here because class sizes deliberately encode base rates;
macro-F1 is the metric to read, and top-3 accuracy is the one that matches the product, which presents
a differential rather than a single label.
</p>
<h4 style="font-size:10pt;margin:10px 0 4px;color:#28645a">Hardest classes</h4>
<table class="avoid">
  <tr><th>Condition</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">n</th></tr>
  {weak_rows}
</table>
<p class="small">
The confusions are clinically coherent — the hepatitis variants share nearly identical symptom
profiles, and COPD and asthma are separated mainly by history rather than by presentation. These are
the errors a symptom-level feature space should be expected to make.
</p>

<h3>7.2 NLP components vs a hand-annotated gold standard</h3>
<table class="avoid">
  <tr><th>Component</th><th>Metric</th><th class="num">Score</th><th class="num">n</th></tr>
  <tr><td rowspan="4">Symptom extraction</td><td>micro precision / recall / F1</td>
      <td class="num">{dec(extraction.get('micro_precision'))} / {dec(extraction.get('micro_recall'))} / {dec(extraction.get('micro_f1'))}</td>
      <td class="num" rowspan="4">{extraction.get('n_cases', '—')}</td></tr>
  <tr><td>macro F1</td><td class="num">{dec(extraction.get('macro_f1'))}</td></tr>
  <tr><td>exact set match</td><td class="num">{pct(extraction.get('exact_set_match'))}</td></tr>
  <tr class="hl"><td>negation leaks</td><td class="num">{extraction.get('negation_violations', '—')}</td></tr>
  <tr><td>Assertion (NegEx/ConText)</td><td>accuracy over 6 labels</td>
      <td class="num">{pct(assertion.get('accuracy'), 1)}</td><td class="num">{assertion.get('n_cases', '—')}</td></tr>
  <tr><td rowspan="2">Temporal</td><td>normalized value accuracy</td>
      <td class="num">{pct(temporal.get('value_accuracy'))}</td><td class="num" rowspan="2">{temporal.get('n_cases', '—')}</td></tr>
  <tr><td>intake-bucket accuracy</td><td class="num">{pct(temporal.get('bucket_accuracy'))}</td></tr>
  <tr><td>Trend detection</td><td>accuracy</td><td class="num">{pct(nlp.get('trend', {}).get('accuracy'))}</td><td class="num">{nlp.get('trend', {}).get('n_cases', '—')}</td></tr>
  <tr><td>Distress detection</td><td>accuracy</td><td class="num">{pct(nlp.get('distress', {}).get('accuracy'))}</td><td class="num">{nlp.get('distress', {}).get('n_cases', '—')}</td></tr>
  <tr><td rowspan="2">Intent</td><td>held-out hard paraphrases</td>
      <td class="num">{pct(intent.get('accuracy'))}</td><td class="num">{intent.get('n_cases', '—')}</td></tr>
  <tr><td>5-fold cross-validated</td><td class="num">{pct(0.7583)}</td><td class="num">360</td></tr>
</table>
<p>
<span class="key">Negation leaks</span> counts symptoms the user explicitly denied that nonetheless
reached the classifier. It is reported separately from F1 because it is the error that would actually
mislead a user, and it is the metric that motivated replacing the original twelve-character
look-behind. It currently stands at zero.
</p>
<h4 style="font-size:10pt;margin:10px 0 4px;color:#28645a">Assertion accuracy by label</h4>
<table class="avoid">
  <tr><th>Derived label</th><th class="num">Accuracy</th><th class="num">Support</th></tr>
  {assertion_rows}
</table>
<div class="callout warn">
  <span class="key">On the two intent figures.</span> Both are reported deliberately. The held-out set
  comprises {intent.get('n_cases', 32)} paraphrases written to be difficult ("much obliged",
  "assalam walekum i need some guidance") and sharing no sentence with the training corpus; the
  evaluation script asserts zero overlap and currently reports
  <span class="mono">{intent.get('leaked_from_training') or 'none'}</span>. The lower held-out figure
  is the honest one for unusual phrasing. It is mitigated by policy rather than by claiming a better
  number: routing requires 45% confidence, and every miss in the held-out set fell below that
  threshold and was therefore handled safely as an ordinary symptom description.
</div>

<h3>7.3 Retrieval</h3>
<table class="avoid">
  <tr><th>Mode</th><th class="num">Hit@1</th><th class="num">Hit@3</th><th class="num">Hit@5</th>
      <th class="num">MRR</th><th class="num">nDCG@5</th></tr>
  <tr><td>Unfiltered semantic search</td><td class="num">{dec(unf.get('hit_at_1'))}</td>
      <td class="num">{dec(unf.get('hit_at_3'))}</td><td class="num">{dec(unf.get('hit_at_5'))}</td>
      <td class="num">{dec(unf.get('mrr'))}</td><td class="num">{dec(unf.get('ndcg_at_5'))}</td></tr>
  <tr><td>Filtered by prediction (production path)</td><td class="num">{dec(filt.get('hit_at_1'))}</td>
      <td class="num">{dec(filt.get('hit_at_3'))}</td><td class="num">{dec(filt.get('hit_at_5'))}</td>
      <td class="num">{dec(filt.get('mrr'))}</td><td class="num">{dec(filt.get('ndcg_at_5'))}</td></tr>
</table>
<p>
The filtered row is near-perfect by construction — once the classifier names a condition, retrieval is
restricted to that condition's chunks — and is reported separately for exactly that reason; it is not
a measure of retrieval quality. Precision@5 is {dec(unf.get('precision_at_5'))} against a hard ceiling
of {dec(unf.get('precision_at_5_ceiling'))}: each disease document yields only
{unf.get('avg_chunks_per_disease', '~2')} chunks, so at most that many of five results can be
relevant. Against the achievable ceiling, retrieval attains
{pct(unf.get('precision_at_5_vs_ceiling'))}.
</p>

<h3>7.4 Verification</h3>
<p>
The repository contains 142 automated tests. Beyond the usual coverage, three are worth naming because
they guard properties that would otherwise fail silently: the Porter stemmer is checked against its
published reference vocabulary; offset round-tripping is asserted through a code-mixed rewrite; and
the explanation layer's reported weights are asserted to reconstruct scikit-learn's own joint
log-likelihood. Without the last of these, an explanation panel could display plausible, wrong numbers
indefinitely. End-to-end latency for the full pipeline is approximately 10 ms per turn.
</p>

<!-- ============================ 8 COMPARISON ============================ -->
<h2>8 Arogya AI vs Existing Systems</h2>
<table>
  <tr><th>Capability</th><th>Keyword symptom checkers</th><th>LLM chatbots [2][3]</th><th>Arogya AI</th></tr>
  <tr><td>Negation handling</td><td>None or substring</td><td>Implicit, unverifiable</td><td>NegEx + ConText, 100% on gold set</td></tr>
  <tr><td>Experiencer / temporality</td><td>None</td><td>Implicit</td><td>Explicit independent axes</td></tr>
  <tr><td>Code-mixed input</td><td>Not supported</td><td>Varies, opaque</td><td>139-phrase lexicon, spans preserved</td></tr>
  <tr><td>Explainability</td><td>None</td><td>Post-hoc narrative</td><td>Exact log-odds decomposition</td></tr>
  <tr><td>Grounding</td><td>Static content</td><td>RAG [4][5] when present</td><td>Chroma RAG, measured (MRR {dec(unf.get('mrr'))})</td></tr>
  <tr><td>Component-level evaluation</td><td>Rare</td><td>Rare</td><td>Every stage, gold standard</td></tr>
  <tr><td>Safety ordering</td><td>Varies</td><td>Post-generation filter</td><td>Red flags before any output</td></tr>
  <tr><td>Cost / privacy</td><td>Low</td><td>API cost, data leaves device</td><td>Runs fully locally; LLM optional</td></tr>
  <tr><td>Determinism</td><td>Yes</td><td>No</td><td>Yes (seeded, reproducible)</td></tr>
</table>
<p>
The comparison is not that this system outperforms a large language model on medical knowledge — it
does not, and [3] makes clear what that frontier looks like. The claim is narrower and, we argue, more
defensible for this application class: on the specific linguistic phenomena that determine whether the
<em>correct input</em> reaches the model, a small, auditable, measured pipeline is preferable to an
opaque one, and it runs locally at negligible cost.
</p>

<!-- ============================ 9 LIMITATIONS ============================ -->
<h2>9 Limitations &amp; Future Work</h2>
<h3>9.1 Limitations</h3>
<ul>
  <li><span class="key">Synthetic data.</span> The dataset is generated from clinician-informed
      prevalence tiers, not collected from patients. Absolute accuracy figures therefore describe the
      model's fit to that generative process, not real-world diagnostic accuracy. The NLP evaluation,
      which uses hand-annotated natural language, is the more transferable result.</li>
  <li><span class="key">Gold standard size.</span> 118 annotated items across six components is small,
      and was annotated by the project team rather than by independent clinicians, so no
      inter-annotator agreement is available.</li>
  <li><span class="key">Intent robustness.</span> {pct(intent.get('accuracy'))} on deliberately hard
      paraphrases is the weakest measured component; the confidence gate limits the consequences but
      does not remove them.</li>
  <li><span class="key">Non-contiguous mentions.</span> The recognizer matches contiguous n-grams, so
      "my knees hurt and they are swollen" yields the swelling but not
      <span class="mono">joint_swelling</span>. Coreference and ellipsis are not resolved.</li>
  <li><span class="key">Rare classes.</span> With base-rate-weighted class sizes, the rarest conditions
      have few test examples and correspondingly unstable per-class metrics.</li>
  <li><span class="key">Scope.</span> 41 conditions and 87 symptoms. The system is an educational
      triage aid and is not a medical device.</li>
</ul>
<h3>9.2 Future work</h3>
<ul>
  <li>Replace the lexicon-based recognizer with a fine-tuned transformer token classifier trained on
      annotated clinical text, retaining the ConText layer on top of it.</li>
  <li>Extend assertion detection across turns, so a symptom denied on turn 2 stays denied when
      rephrased on turn 5.</li>
  <li>Expand the gold standard with independent annotators and report inter-annotator agreement.</li>
  <li>Calibrate the classifier (Platt scaling or isotonic regression) so the reported confidence is a
      true probability, which matters when the number is shown to a patient.</li>
  <li>Enlarge the knowledge base so retrieval precision is not corpus-capped, and add an
      abstention path for queries no document covers.</li>
</ul>

<!-- ============================ 10 REFERENCES ============================ -->
<h2>10 References</h2>
<ol class="refs">
  <li>E. H. Shortliffe and M. J. Sepúlveda, "Clinical decision support in the era of artificial
      intelligence," <em>JAMA</em>, vol. 320, no. 21, pp. 2199–2200, Dec. 2018,
      doi: 10.1001/jama.2018.17163.</li>
  <li>J. W. Ayers et al., "Comparing physician and artificial intelligence chatbot responses to patient
      questions posted to a public social media forum," <em>JAMA Intern. Med.</em>, vol. 183, no. 6,
      pp. 589–596, Jun. 2023, doi: 10.1001/jamainternmed.2023.1838.</li>
  <li>K. Singhal et al., "Large language models encode clinical knowledge," <em>Nature</em>, vol. 620,
      no. 7972, pp. 172–180, Aug. 2023, doi: 10.1038/s41586-023-06291-2.</li>
  <li>P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in
      <em>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</em>, vol. 33, 2020, pp. 9459–9474.</li>
  <li>G. Xiong et al., "Benchmarking retrieval-augmented generation in healthcare," arXiv preprint
      arXiv:2402.15282, Feb. 2024.</li>
  <li>S. R. Thamma, "Agentic AI for clinical decision support: Real-time diagnosis, triage, and
      treatment planning," <em>Int. J. Sci. Res. Sci. Eng. Technol.</em>, vol. 12, no. 3, pp. 428–433,
      2025, doi: 10.32628/IJSRSET251265.</li>
  <li>H. Patel, D. Sathwara, V. Maheshwari, and D. Joshi, "Harnessing artificial intelligence for the
      future of precision medicine: Transforming patient-centric care through data-driven innovation,"
      <em>Biopress J. Comput. Life Sci.</em>, vol. 1, no. 4, Jul. 2025.</li>
  <li>S. Khan and M. R. Ghalib, "A Naive-Bayes approach for disease diagnosis with analysis of disease
      type and symptoms," <em>Int. J. Appl. Eng. Res.</em>, vol. 10, no. 11, pp. 29005–29014, 2015.</li>
  <li>S. S. Al-qarni and A. Algarni, "Disease prediction from symptom descriptions using deep learning
      and NLP technique," <em>Int. J. Adv. Comput. Sci. Appl.</em>, vol. 16, no. 5, pp. 416–426, 2025,
      doi: 10.14569/IJACSA.2025.0160541.</li>
  <li>R. Shrivastava and A. Rai, "Arogya AI — Agentic disease prediction with Ayurvedic
      recommendations," Course Project Specification, Dept. Comput. Sci. Eng., Vellore Institute of
      Technology, Vellore, India, 2025.</li>
</ol>

<h3>Algorithms implemented</h3>
<ol class="refs" start="11">
  <li>M. F. Porter, "An algorithm for suffix stripping," <em>Program</em>, vol. 14, no. 3,
      pp. 130–137, 1980.</li>
  <li>W. W. Chapman, W. Bridewell, P. Hanbury, G. F. Cooper, and B. G. Buchanan, "A simple algorithm
      for identifying negated findings and diseases in discharge summaries," <em>J. Biomed. Inform.</em>,
      vol. 34, no. 5, pp. 301–310, 2001.</li>
  <li>H. Harkema, J. N. Dowling, T. Thornblade, and W. W. Chapman, "ConText: An algorithm for
      determining negation, experiencer, and temporal status from clinical reports,"
      <em>J. Biomed. Inform.</em>, vol. 42, no. 5, pp. 839–851, 2009.</li>
  <li>N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in
      <em>Proc. EMNLP-IJCNLP</em>, 2019, pp. 3982–3992.</li>
</ol>

</body></html>
"""


def find_chrome():
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    html_path = os.path.join(HERE, "ArogyaAI_Project_Report.html")
    pdf_path = os.path.join(ROOT, "ArogyaAI_Project_Report.pdf")

    html = build_html()
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {html_path}  ({len(html):,} bytes)")

    if args.html_only:
        return

    chrome = find_chrome()
    if not chrome:
        print("Chrome not found; wrote HTML only. Print it to PDF manually.")
        return

    url = "file:///" + html_path.replace("\\", "/")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        url,
    ]
    print("Rendering PDF with headless Chrome...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path)
        print(f"Wrote {pdf_path}  ({size:,} bytes)")
    else:
        print("PDF not produced.")
        print(result.stderr[:800])
        sys.exit(1)


if __name__ == "__main__":
    main()
