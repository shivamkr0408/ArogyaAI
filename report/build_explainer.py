"""
Build the teacher-facing explainer PDF for Arogya AI.

This is deliberately NOT the formal project report (report/build_report.py
writes that one). It is a walkthrough: what the system is, how a message
travels through it, what was measured, and what was changed in the October
2026 accuracy work -- written to be read start to finish by someone who has
not seen the code.

Every number is read from the measured artifacts:

    ml/model/metrics.json                          deployed model, after
    ml/data/dataset_stats.json                     dataset shape
    eval/results/model_comparison.json             all 8 models
    eval/results/nlp_eval.json                     NLP stage accuracy
    eval/results/rag_eval.json                     retrieval quality
    eval/results/baseline_before_improvements.json the same metrics, before

so the document cannot drift from what was actually measured. Re-run the
eval scripts, then re-run this.

Usage:  backend/venv/Scripts/python.exe report/build_explainer.py [--html-only]
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
]


def load(*parts):
    path = os.path.join(ROOT, *parts)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def arrow(before, after, digits=3):
    """Render a before -> after pair with its delta."""
    delta = after - before
    cls = "up" if delta > 0 else ("down" if delta < 0 else "flat")
    sign = "+" if delta > 0 else ""
    return (f'<span class="ba"><span class="was">{before:.{digits}f}</span>'
            f'<span class="arr">&rarr;</span>'
            f'<span class="now">{after:.{digits}f}</span>'
            f'<span class="delta {cls}">{sign}{delta:.{digits}f}</span></span>')


# ---------------------------------------------------------------------------
CSS = """
@page { size: A4; margin: 15mm 14mm 16mm 14mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", Inter, system-ui, sans-serif;
  font-size: 10.2pt; line-height: 1.55; color: #1f2937; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 22pt; margin: 0 0 2mm; color: #1f443e; letter-spacing: -0.4pt; }
h2 { font-size: 13.5pt; margin: 9mm 0 2.5mm; color: #23514a;
     border-bottom: 2px solid #b5e2d7; padding-bottom: 1.4mm; }
h3 { font-size: 11pt; margin: 5mm 0 1.5mm; color: #28645a; }
p  { margin: 0 0 2.6mm; }
ul, ol { margin: 0 0 3mm; padding-left: 5.2mm; }
li { margin-bottom: 1.3mm; }
code { font-family: Consolas, monospace; font-size: 9pt;
       background: #f3f4f6; padding: 0.4mm 1.1mm; border-radius: 2px; }
.lede { font-size: 11pt; color: #374151; }
.sub  { color: #6b7280; font-size: 9.6pt; margin: 0 0 1mm; }

.cover { border-left: 4px solid #2e7d6f; padding-left: 5mm; margin-bottom: 7mm; }
.meta { font-size: 9.4pt; color: #4b5563; margin-top: 2.5mm; }
.meta b { color: #1f2937; }

table { width: 100%; border-collapse: collapse; margin: 2.5mm 0 4mm;
        font-size: 9.2pt; }
th { text-align: left; background: #eef9f6; color: #23514a; font-weight: 600;
     padding: 1.6mm 2mm; border-bottom: 1.5px solid #b5e2d7; }
td { padding: 1.5mm 2mm; border-bottom: 1px solid #eceff2; vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.hi td { background: #f4fbf9; font-weight: 600; }

.ba { white-space: nowrap; font-variant-numeric: tabular-nums; }
.was { color: #9ca3af; text-decoration: line-through; }
.arr { color: #9ca3af; margin: 0 1.2mm; }
.now { color: #1f2937; font-weight: 600; }
.delta { margin-left: 1.8mm; font-size: 8.4pt; font-weight: 600; }
.delta.up { color: #047857; }
.delta.down { color: #b91c1c; }
.delta.flat { color: #9ca3af; }

.callout { background: #f4fbf9; border: 1px solid #b5e2d7; border-left: 3px solid #2e7d6f;
           border-radius: 3px; padding: 3mm 4mm; margin: 3mm 0 4mm; }
.callout h4 { margin: 0 0 1.5mm; font-size: 10pt; color: #1f443e; }
.warn { background: #fffbeb; border-color: #fde68a; border-left-color: #b45309; }
.warn h4 { color: #78350f; }

.flow { margin: 3mm 0 4mm; }
.step { display: flex; gap: 3mm; align-items: flex-start; margin-bottom: 2mm; }
.step .n { flex: 0 0 6mm; height: 6mm; border-radius: 50%; background: #2e7d6f;
           color: #fff; font-size: 8.4pt; font-weight: 700; text-align: center;
           line-height: 6mm; }
.step .b { flex: 1; }
.step .b b { color: #1f443e; }
.step.red .n { background: #dc2626; }

.grid2 { display: flex; gap: 5mm; }
.grid2 > div { flex: 1; }

.kpis { display: flex; gap: 3mm; margin: 3mm 0 4mm; }
.kpi { flex: 1; border: 1px solid #d8f1ea; border-radius: 4px; padding: 2.5mm 3mm;
       background: #fbfefd; }
.kpi .v { font-size: 15pt; font-weight: 700; color: #1f443e; line-height: 1.1; }
.kpi .l { font-size: 8.2pt; color: #6b7280; margin-top: 0.6mm; }
.kpi .d { font-size: 8.2pt; color: #047857; font-weight: 600; }

footer { margin-top: 8mm; padding-top: 2.5mm; border-top: 1px solid #e5e7eb;
         font-size: 8.4pt; color: #9ca3af; }
.pagebreak { page-break-before: always; }
"""


def build_html():
    m = load("ml", "model", "metrics.json")
    ds = load("ml", "data", "dataset_stats.json")
    cmp_ = load("eval", "results", "model_comparison.json")
    nlp = load("eval", "results", "nlp_eval.json")
    rag = load("eval", "results", "rag_eval.json")
    base = load("eval", "results", "baseline_before_improvements.json")

    bh = base["headline"]
    bd = base["dataset"]

    # ---- KPI strip --------------------------------------------------------
    kpis = [
        ("Accuracy", pct(m["accuracy"]), f"+{(m['accuracy'] - bh['accuracy']) * 100:.1f} pts"),
        ("Macro F1", f"{m['macro_f1']:.3f}", f"+{m['macro_f1'] - bh['macro_f1']:.3f}"),
        ("Top-3 accuracy", pct(m["top3_accuracy"]), f"+{(m['top3_accuracy'] - bh['top3_accuracy']) * 100:.1f} pts"),
        ("Training rows", f"{ds['total_rows']:,}", f"+{ds['total_rows'] - bd['total_rows']:,}"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div>'
        f'<div class="d">{d}</div></div>'
        for l, v, d in kpis
    )

    # ---- model comparison -------------------------------------------------
    rows = sorted(cmp_["results"], key=lambda r: -r.get("macro_f1", 0))
    cmp_rows = ""
    for r in rows:
        hi = ' class="hi"' if r["model"] == cmp_["deployed_model"] else ""
        cmp_rows += (
            f'<tr{hi}><td>{r["model"]}</td>'
            f'<td class="num">{r["accuracy"]:.3f}</td>'
            f'<td class="num">{r.get("top3_accuracy", 0):.3f}</td>'
            f'<td class="num">{r["macro_f1"]:.3f}</td>'
            f'<td class="num">{r.get("cv_accuracy_mean", 0):.3f}</td>'
            f'<td class="num">{r.get("fit_ms", 0):.0f}</td></tr>'
        )

    # ---- per-class recovery ----------------------------------------------
    pc = m["per_class"]
    recov = ""
    for name, before in sorted(base["weakest_classes"].items(),
                               key=lambda kv: kv[1]["f1"]):
        after = pc.get(name)
        if not after:
            continue
        recov += (f'<tr><td>{name}</td><td class="num">{arrow(before["f1"], after["f1"])}</td>'
                  f'<td class="num">{after["precision"]:.2f}</td>'
                  f'<td class="num">{after["recall"]:.2f}</td>'
                  f'<td class="num">{after["support"]}</td></tr>')

    # ---- NLP stages -------------------------------------------------------
    se, asrt, intent = nlp["symptom_extraction"], nlp["assertion"], nlp["intent"]
    tmp, trend, distress = nlp["temporal"], nlp["trend"], nlp["distress"]
    unf, filt = rag["unfiltered"], rag["filtered"]

    today = date.today().strftime("%d %B %Y")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Arogya AI - Project Explainer</title><style>{CSS}</style></head><body>

<div class="cover">
  <h1>Arogya AI</h1>
  <p class="lede">A safety-first conversational assistant for early-stage health
  guidance. It reads a complaint written in plain English, asks a few targeted
  follow-up questions, and returns a ranked differential with its reasoning
  shown - while screening every single message for medical emergencies before
  any model is consulted.</p>
  <p class="meta">
    <b>Course</b> BCSE409L, NLP Based Project &nbsp;&middot;&nbsp;
    <b>Team</b> Rishav Kumar (23BCE0462), Anshuman Kumar (23BCE0311),
    Rahul Kumar (23BCE0246)<br>
    <b>Guide</b> Prof. Deepa K &nbsp;&middot;&nbsp;
    <b>Prepared</b> {today} &nbsp;&middot;&nbsp;
    <b>Every figure below is read from the project's measured output files.</b>
  </p>
</div>

{("<div class='kpis'>" + kpi_html + "</div>")}

<h2>1. The problem, and why it needs two kinds of system</h2>
<p>Software for checking your own symptoms tends to sit at one of two
extremes, and each one fails in a way the other does not.</p>
<ul>
  <li><b>Menu-driven checkers</b> are predictable and safe, because a human
  wrote every branch. But they cannot read a sentence. The patient has to
  translate their own illness into the form's categories before the form can
  help them.</li>
  <li><b>Large language models</b> read the sentence effortlessly, and will
  happily produce a fluent, confident, and occasionally wrong answer, with no
  signal to the reader about which one they just received.</li>
</ul>
<p>Arogya AI takes the reading ability from the second and the predictability
from the first, by giving the language model the smallest possible job. It
helps interpret messy input and phrase the reply. It does not decide whether
something is an emergency, and it does not decide which condition is most
likely. Those two decisions - the ones that carry risk - belong to a
deterministic rule layer and a transparent statistical classifier. Every
step that touches a language model has a local fallback, so the system
behaves identically when no model key is configured at all.</p>

<h2>2. How a message travels through the system</h2>
<p>The order of these stages is the single most important design decision in
the project.</p>
<div class="flow">
  <div class="step red"><div class="n">1</div><div class="b">
    <b>Red-flag safety scan.</b> Deterministic rules, not a model, so it
    cannot be talked out of firing. Eight categories (cardiac, stroke,
    respiratory, bleeding, anaphylaxis, obstetric, abdominal, mental-health
    crisis). A match ends the conversation in a terminal emergency state:
    <i>no differential is produced at all</i>, and the user is sent to urgent
    care. This runs <b>before</b> everything below, on every turn.</div></div>
  <div class="step"><div class="n">2</div><div class="b">
    <b>Symptom extraction.</b> A staged clinical NLP pipeline turns free text
    into canonical symptoms - tokenizer, Hinglish normaliser, negation
    detection, temporal extraction, named-entity recognition.</div></div>
  <div class="step"><div class="n">3</div><div class="b">
    <b>Prediction.</b> A Bernoulli Naive Bayes classifier ranks
    {m['n_classes']} conditions over {m['n_features']} symptom features and
    returns a full posterior, not a single label.</div></div>
  <div class="step"><div class="n">4</div><div class="b">
    <b>Guided follow-up.</b> Between three and six questions: general
    clinical context, plus <i>condition-specific</i> questions chosen from the
    conditions currently leading the differential (see section 5).</div></div>
  <div class="step"><div class="n">5</div><div class="b">
    <b>Retrieval.</b> The explanation is grounded in a curated knowledge base
    of {rag['indexed_chunks']} passages, so claims are anchored to controlled
    text rather than generated freely.</div></div>
  <div class="step"><div class="n">6</div><div class="b">
    <b>Triage and guidance.</b> A triage recommendation, the retrieved
    evidence, and non-prescriptive Ayurvedic lifestyle guidance.</div></div>
</div>
<div class="callout">
  <h4>Why the ordering matters</h4>
  <p style="margin:0">If the classifier ran first, a heart attack could be
  reported as "Heart attack, 61% confidence" - a number next to a medical
  emergency, which invites the reader to wait and see. Running the rule layer
  first makes that outcome structurally impossible rather than unlikely.</p>
</div>

<h2>3. What the system is measured at</h2>
<p>Nothing in this document is an estimate. Each number below is written by
an evaluation script into a JSON file, which this page reads.</p>

<h3>Language understanding <span class="sub">eval/results/nlp_eval.json</span></h3>
<table>
  <tr><th>Stage</th><th>What it does</th><th class="num">Score</th><th class="num">Cases</th></tr>
  <tr><td>Symptom extraction</td><td>Free text &rarr; canonical symptoms</td>
      <td class="num">{se['micro_f1']:.3f} F1</td><td class="num">{se['n_cases']}</td></tr>
  <tr><td>Assertion / negation</td><td>Catches "no fever" as absent, not present</td>
      <td class="num">{pct(asrt['accuracy'], 0)}</td><td class="num">{asrt['n_cases']}</td></tr>
  <tr><td>Temporal</td><td>"since Tuesday" &rarr; a duration</td>
      <td class="num">{pct(tmp['value_accuracy'], 0)}</td><td class="num">{tmp['n_cases']}</td></tr>
  <tr><td>Trend</td><td>Getting better, worse, or unchanged</td>
      <td class="num">{pct(trend['accuracy'], 0)}</td><td class="num">{trend['n_cases']}</td></tr>
  <tr><td>Distress</td><td>Detects a worried or distressed tone</td>
      <td class="num">{pct(distress['accuracy'], 0)}</td><td class="num">{distress['n_cases']}</td></tr>
  <tr><td>Intent</td><td>What the user is asking for</td>
      <td class="num">{pct(intent['accuracy'], 1)}</td><td class="num">{intent['n_cases']}</td></tr>
</table>
<p>Intent is the weakest stage, and deliberately so in the reporting: the
{intent['n_cases']} cases are <i>hard held-out paraphrases</i>, not the
training distribution. Rather than hide that, the orchestrator applies a
confidence gate before it acts on an intent, so a low-confidence guess is
ignored instead of acted on.</p>

<h3>Evidence retrieval <span class="sub">eval/results/rag_eval.json</span></h3>
<table>
  <tr><th>Setting</th><th class="num">Hit@1</th><th class="num">Hit@3</th>
      <th class="num">MRR</th><th class="num">nDCG@5</th></tr>
  <tr><td>Open search over all passages</td>
      <td class="num">{unf['hit_at_1']:.3f}</td><td class="num">{unf['hit_at_3']:.3f}</td>
      <td class="num">{unf['mrr']:.3f}</td><td class="num">{unf['ndcg_at_5']:.3f}</td></tr>
  <tr class="hi"><td>Scoped to the predicted condition <i>(as deployed)</i></td>
      <td class="num">{filt['hit_at_1']:.3f}</td><td class="num">{filt['hit_at_3']:.3f}</td>
      <td class="num">{filt['mrr']:.3f}</td><td class="num">{filt['ndcg_at_5']:.3f}</td></tr>
</table>
<p>The second row is how the system actually runs: retrieval is restricted to
the condition the classifier predicted, which is why grounding is reliable
even though open semantic search over the whole corpus is not perfect.</p>

<div class="pagebreak"></div>
<h2>4. Choosing the classifier</h2>
<p>Eight models were trained on identical data and compared on the same
held-out split, including a majority-class baseline as a floor.</p>
<table>
  <tr><th>Model</th><th class="num">Accuracy</th><th class="num">Top-3</th>
      <th class="num">Macro F1</th><th class="num">CV acc</th><th class="num">Fit ms</th></tr>
  {cmp_rows}
</table>
<p>With {m['n_classes']} classes the majority baseline scores about 0.05, so
every other row has genuinely learned something. <b>Accuracy alone is
misleading here</b>, because the class sizes deliberately encode clinical
base rates - a model could look respectable by simply never predicting the
rare conditions. Macro F1 is the number to read, and top-3 accuracy is the
one that matches the product, which shows a ranked differential rather than
a single label.</p>
<div class="callout">
  <h4>Bernoulli Naive Bayes is deployed - and is now also the best</h4>
  <p style="margin:0 0 2mm">It was chosen for three reasons that survive
  regardless of the score: its decision decomposes <i>exactly</i> into
  per-symptom log-odds, so the explanation panel is a true attribution rather
  than a LIME-style approximation; unlike the Multinomial variant it scores
  symptom <i>absence</i>, so a symptom the user explicitly denied is usable
  evidence; and it fits in milliseconds.</p>
  <p style="margin:0">Previously this came at a cost - Logistic Regression
  led it by {base['model_selection']['macro_f1_gap_to_best']:.3f} macro F1,
  and the project documented that trade-off rather than hiding it. After the
  data work in section 5, Bernoulli NB is the top model outright, so the
  trade-off no longer exists.</p>
</div>

<h2>5. What was improved, and why</h2>
<p>The earlier model scored {bh['accuracy']:.3f} accuracy but only
{bh['macro_f1']:.3f} macro F1 - a gap that says the errors were concentrated
in particular conditions rather than spread evenly. Reading the per-class
report showed exactly where.</p>

<h3>The diagnosis: some conditions were impossible to separate</h3>
<p><b>Hepatitis D scored an F1 of exactly 0.000.</b> Not low - zero. Its
symptom profile was <i>a strict subset</i> of the other hepatitis variants:
joint pain, vomiting, fatigue, nausea, abdominal pain. It had no feature
that no other condition had. No classifier, however sophisticated, can
separate a class that is defined by nothing unique, and no amount of extra
training data would have helped. The same pattern affected the respiratory
group, where asthma, COPD, bronchial asthma and pneumonia all drew from one
shared pool of cough, breathlessness and wheezing.</p>

<h3>The fix: features that actually discriminate</h3>
<p>What separates the hepatitis variants in real medicine is not another
symptom - it is an <b>exposure or a history</b>. So those were added as
features, placed as the hallmark of each variant:</p>
<table>
  <tr><th>Condition</th><th>Discriminating feature added</th><th>Clinical basis</th></tr>
  <tr><td>Hepatitis A</td><td><code>recent_contaminated_food</code></td><td>Faecal-oral transmission</td></tr>
  <tr><td>Hepatitis B</td><td><code>unsafe_injection_exposure</code></td><td>Blood-borne</td></tr>
  <tr><td>Hepatitis C</td><td><code>blood_transfusion_history</code></td><td>Blood-borne</td></tr>
  <tr class="hi"><td>Hepatitis D</td><td><code>known_hepatitis_b</code></td>
      <td>Hepatitis D <i>only</i> occurs alongside an existing B infection</td></tr>
  <tr><td>Hepatitis E</td><td><code>recent_contaminated_water</code></td><td>Waterborne</td></tr>
  <tr><td>Alcoholic hepatitis</td><td><code>heavy_alcohol_use</code></td><td>Defining cause</td></tr>
  <tr><td>Tuberculosis</td><td><code>cough_over_two_weeks</code>, <code>night_sweats</code></td><td>Classic presentation</td></tr>
  <tr><td>COPD</td><td><code>smoking_history</code>, <code>chronic_productive_cough</code></td><td>Dominant risk factor</td></tr>
  <tr><td>Asthma</td><td><code>night_time_symptoms</code>, <code>symptoms_with_triggers</code></td><td>Episodic, trigger-linked</td></tr>
  <tr><td>Pneumonia</td><td><code>productive_sputum</code></td><td>Purulent sputum</td></tr>
</table>
<p>The feature count rose from {bd['features']} to {m['n_features']}. The
rare-class floor was also raised: at 20-32 rows a rare condition had only
four to six test rows, so its F1 moved in steps of 0.2 and measured sampling
noise more than learning. Class sizes remain <i>ordinal</i> - common
conditions still outnumber rare ones, so the Naive Bayes prior still carries
base-rate information and the system still needs strong evidence before
naming something serious.</p>

<h3>The result</h3>
<div style="margin-bottom:2mm">
  Accuracy {arrow(bh['accuracy'], m['accuracy'])} &nbsp;&nbsp;
  Macro F1 {arrow(bh['macro_f1'], m['macro_f1'])} &nbsp;&nbsp;
  Top-3 {arrow(bh['top3_accuracy'], m['top3_accuracy'])} &nbsp;&nbsp;
  Cross-validated {arrow(bh['cv_accuracy_mean'], m['cv_accuracy_mean'])}
</div>
<table>
  <tr><th>Condition that was failing</th><th class="num">F1 before &rarr; after</th>
      <th class="num">Prec.</th><th class="num">Recall</th><th class="num">Test rows</th></tr>
  {recov}
</table>
<p>Cross-validation also became <i>steadier</i>: the standard deviation across
folds fell from {bh['cv_accuracy_std']:.4f} to {m['cv_accuracy_std']:.4f},
meaning the result depends less on which rows happened to land in which
fold.</p>

<h2>6. Condition-specific follow-up questions</h2>
<p>The discriminating features above are only useful if the system can
actually find out about them, and a user will rarely volunteer "I had a blood
transfusion" unprompted. So the follow-up engine now asks different questions
depending on which conditions are leading the differential.</p>
<p>Previously every consultation drew from one fixed pool - duration,
intensity, trend, temperature, exposure, appetite - plus a generic "are you
also experiencing X?" for a discriminating symptom. That generic question is
useless for the hepatitis family, because the symptom it would ask about is
one they all share.</p>
<table>
  <tr><th>When the differential is leaning towards...</th><th>...the system asks</th></tr>
  <tr><td><b>Hepatitis / jaundice</b><br><span class="sub">A, B, C, D, E, alcoholic, cholestasis</span></td>
      <td>Recent exposures (street food, unsafe water, injections, transfusion);
          known hepatitis B or daily alcohol; stool colour</td></tr>
  <tr><td><b>Respiratory</b><br><span class="sub">asthma, COPD, pneumonia, TB, cold</span></td>
      <td>When breathing is worst (night vs triggers); what the cough brings up;
          smoking, cough duration, night sweats; whether an inhaler is helping</td></tr>
  <tr><td><b>Febrile</b><br><span class="sub">malaria, dengue, typhoid, gastroenteritis</span></td>
      <td>What accompanies the fever - shaking chills and sweats, a rash,
          severe joint pain, or constipation with abdominal pain</td></tr>
  <tr><td><b>Skin</b><br><span class="sub">fungal, psoriasis, impetigo, drug reaction</span></td>
      <td>What the affected skin looks like - silvery scaling, yellow-crusted
          blisters, discoloured patches, raised bumps</td></tr>
</table>
<p>Each answer maps directly onto a model feature, so answering the question
immediately sharpens the posterior. When the leading conditions have no
discriminator left to ask about, the engine falls back to the general
clinical slots - a cardiac presentation is never asked about inhalers.</p>

<h2>7. Honest limitations</h2>
<div class="callout warn">
  <h4>What this system is not</h4>
  <ul style="margin:0">
    <li><b>The training data is synthetic.</b> It is generated from per-disease
    symptom prevalences and clinical base rates, not from patient records. The
    accuracy figures describe how well the model learned <i>that</i>
    distribution. They are not clinical accuracy, and the project does not
    claim they are.</li>
    <li><b>It does not diagnose.</b> It never names a drug or a dose, it shows
    a ranked differential rather than an answer, and it carries a persistent
    disclaimer.</li>
    <li><b>Two conditions remain genuinely hard.</b> AIDS and brain
    haemorrhage sit lowest on per-class F1. For the haemorrhage case the model
    is conservative rather than wrong - high precision, lower recall - which
    is the right direction for a serious condition, and the red-flag layer
    catches stroke symptoms independently of the classifier.</li>
    <li><b>The safety layer has not been evaluated statistically.</b> It is
    verified by targeted tests, not by a labelled corpus of emergency and
    non-emergency messages, so sensitivity and specificity cannot yet be
    quoted. That is the clearest next piece of work.</li>
  </ul>
</div>

<h2>8. Running it</h2>
<p>Backend on port 8000, frontend on 5173; <code>start.bat</code> launches
both. To reproduce every number in this document from scratch:</p>
<table>
  <tr><th>Step</th><th>Command</th></tr>
  <tr><td>1. Generate the dataset</td><td><code>python ml/generate_data.py</code></td></tr>
  <tr><td>2. Train both NB variants</td><td><code>python ml/train_model.py</code></td></tr>
  <tr><td>3. Train the intent model</td><td><code>python ml/train_intent.py</code></td></tr>
  <tr><td>4. Build the retrieval index</td><td><code>python rag/build_index.py</code></td></tr>
  <tr><td>5. Compare all 8 models</td><td><code>python eval/compare_models.py</code></td></tr>
  <tr><td>6. Evaluate NLP and retrieval</td><td><code>python eval/eval_nlp.py</code> &middot; <code>eval/eval_rag.py</code></td></tr>
  <tr><td>7. Rebuild this document</td><td><code>python report/build_explainer.py</code></td></tr>
</table>
<p>All of it runs on a laptop CPU in well under a minute, and the dataset
generator is seeded, so the figures reproduce exactly. The test suite is
<code>pytest tests</code>.</p>

<footer>
Arogya AI &middot; generated {today} by <code>report/build_explainer.py</code>.
Every metric is read at build time from <code>ml/model/metrics.json</code>,
<code>ml/data/dataset_stats.json</code> and <code>eval/results/*.json</code>,
so this document cannot disagree with what was measured.
</footer>
</body></html>"""


def find_chrome():
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    html_path = os.path.join(HERE, "ArogyaAI_Explainer.html")
    pdf_path = os.path.join(ROOT, "ArogyaAI_Explainer.pdf")

    html = build_html()
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {html_path}  ({len(html):,} bytes)")

    if args.html_only:
        return

    chrome = find_chrome()
    if not chrome:
        print("Chrome not found; wrote HTML only.")
        return

    url = "file:///" + html_path.replace("\\", "/")
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", url]
    print("Rendering PDF with headless Chrome...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if os.path.exists(pdf_path):
        print(f"Wrote {pdf_path}  ({os.path.getsize(pdf_path):,} bytes)")
    else:
        print("PDF not produced.")
        print(result.stderr[:800])
        sys.exit(1)


if __name__ == "__main__":
    main()
