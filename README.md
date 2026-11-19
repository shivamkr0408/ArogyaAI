# Arogya AI

A conversational, NLP-driven **early-stage health-guidance** web app. Users describe symptoms in plain English — or in code-mixed Hinglish — in a chat interface. The system normalizes the text, recognizes clinical entities with their character spans, decides for each one whether it is actually being *reported* (as opposed to denied, hypothesised, or attributed to a relative), checks emergency red flags on every turn, predicts likely conditions with a Naive Bayes classifier, explains that prediction symptom by symptom, grounds the explanation in a local RAG knowledge base, and pairs it with curated Ayurvedic lifestyle guidance.

> **This is an educational, informational tool and not a medical diagnosis. Always consult a licensed healthcare professional.**

**Course project:** VIT, BCSE409L — NLP Based Project.

---

## What makes this an NLP project

The language layer is the substance here, not a thin wrapper over a classifier. Every stage is a named algorithm, implemented from scratch, separately testable, and separately **measured** (see [Evaluation](#evaluation)).

| Stage | Algorithm | What it fixes |
|---|---|---|
| **Normalization** | Unicode NFC · contraction expansion · elongation collapsing · code-mixed rewriting · bounded Damerau-Levenshtein spell repair | `"mujhe tez bukhar hai"`, `"coughhh"`, `"feaver"`, `"don't"` all become clean clinical English — while an **offset map** preserves where every word came from |
| **Tokenization** | Offset-preserving tokenizer + full **Porter (1980)** stemmer | `aching`/`aches`/`ached` collapse to one lookup key; spans survive for highlighting |
| **Entity recognition** | Three-tier cascade: longest-match dictionary → rapidfuzz token ratio → MiniLM cosine similarity | Each entity reports *which* strategy found it and with what confidence |
| **Assertion** | **NegEx** (Chapman 2001) extended to **ConText** (Harkema 2009) | `"no cough but I do have a fever"`, `"my father has diabetes"`, `"I had dengue last year"`, `"if I get a fever"` — all handled correctly on three independent axes |
| **Temporal** | TIMEX-style duration / frequency / trend normalization | `"since 3 days"`, `"teen din se"` resolve to days, so the bot stops asking what you already said |
| **Intent** | TF-IDF word + char n-grams → logistic regression, with high-precision regex rules first | `"what should I eat?"` after a result is answered, not treated as a new symptom |
| **Distress** | Weighted lexicon with negation and intensifier scaling | An anxious message gets one sentence of acknowledgement before the clinical content |
| **Explainability** | Exact Naive Bayes log-odds decomposition | Per-symptom attribution that sums back to the model's own score — not a LIME/SHAP approximation |

### The three ConText axes

A symptom only reaches the classifier when it is **affirmed**, **recent**, and about the **patient**:

```
"my father never had asthma"
   negation    = negated      <- never
   experiencer = other        <- my father
   temporality = recent
   => not a reported symptom (on two independent grounds)
```

This replaced a 12-character look-behind that got every one of the above wrong.

---

## Architecture

```
┌──────────────────┐         ┌──────────────────────────────────────────────┐
│  React Frontend  │  /api   │  FastAPI Backend                             │
│  (Vite + TS)     │ ──────► │  Orchestrator (state machine + dialogue       │
│  port 5173       │         │                policy w/ intent confidence)   │
│                  │         │   │                                           │
│  · Chat          │         │   ├─ app/nlp/  ── the pipeline ──┐            │
│  · NLP Lab       │         │   │    normalizer → tokenizer     │           │
│  · Evaluation    │         │   │    → ner → negation           │           │
│  · Explanation   │         │   │    → temporal → intent        │           │
└──────────────────┘         │   │    → sentiment → pipeline     │           │
                             │   ├─ Safety layer (red flags, runs first)     │
                             │   ├─ Naive Bayes predictor (Bernoulli)        │
                             │   ├─ Explainability (exact log-odds)          │
                             │   ├─ Chroma RAG retriever                     │
                             │   └─ Ayurveda KB lookup                       │
                             └──────────────────────────────────────────────┘
                                 │          │         │          │
                          SQLite │  ml/model│  rag/   │  .env LLM│
                        (sessions) (joblib)  (chroma)  (optional)
```

The LLM is **optional**. Without an API key the pipeline runs entirely locally and the app works end to end; with a key, the LLM's symptom list is *unioned* with the local pipeline's and the assertion layer still has final say (an LLM will happily return "fever" for "I don't have a fever").

---

## Project structure

```
backend/app/
  nlp/                    THE NLP PIPELINE
    lexicon.py            canonical vocab, colloquial aliases, Hinglish, diseases
    tokenizer.py          offset-preserving tokenizer + Porter stemmer
    normalizer.py         4 normalization passes with an offset alignment table
    negation.py           NegEx + ConText, 3 assertion axes
    temporal.py           duration / frequency / trend normalization
    ner.py                exact → fuzzy → embedding recognition cascade
    intent.py             TF-IDF + logistic regression (+ regex rules)
    intent_corpus.py      360-example training corpus
    sentiment.py          distress lexicon
    pipeline.py           composes all stages, with per-stage timings
  services/
    orchestrator.py       state machine, dialogue policy, result assembly
    explain.py            exact Naive Bayes decomposition
    predictor.py          model loading (AROGYA_MODEL selects the variant)
    followup.py           structured clinical intake
    retriever.py          Chroma retrieval with similarity scores
  routes/
    chat · session · health · debug · report · stats · nlp
  safety/red_flags.py     emergency patterns, run before anything is shown
ml/
  generate_data.py        prevalence-weighted synthetic dataset w/ base rates
  train_model.py          trains BOTH NB variants, writes metrics.json
  train_intent.py         fits + cross-validates the intent classifier
eval/
  compare_models.py       8 classifiers, identical splits and folds
  eval_nlp.py             NLP components vs a hand-annotated gold standard
  eval_rag.py             Hit@k / MRR / nDCG for retrieval
  gold/                   nlp_gold.json, rag_gold.json
  results/                JSON the API serves and the dashboard renders
frontend/src/components/
  NlpLabPanel.tsx         live pipeline inspector
  EvaluationPanel.tsx     measured results dashboard
  ExplanationCard.tsx     per-symptom log-odds (diverging bars)
  EntityHighlight.tsx     entity spans rendered over the user's own text
tests/                    142 tests
```

---

## Prerequisites

- Python **3.11 or 3.12** (*not 3.13* — `chromadb` has no prebuilt Windows wheel for it)
- Node.js 18+ / npm
- *(Optional)* an LLM API key in `backend/.env`

## Quick start

```bash
# 1. Backend
cd backend
python -m venv venv
venv\Scripts\activate              # Windows
pip install -r requirements.txt

# 2. Build the artifacts (once)
python ..\ml\generate_data.py      # prevalence-weighted dataset
python ..\ml\train_model.py        # both Naive Bayes variants + metrics.json
python ..\ml\train_intent.py       # intent classifier
python ..\rag\build_index.py       # Chroma vector store

# 3. Run
python -m uvicorn app.main:app --port 8001
```

```bash
# 4. Frontend (separate terminal)
cd frontend
npm install
npm run dev                        # http://localhost:5173
```

Or just run `start.bat` from the repo root to launch both.

Verify: `http://localhost:8001/api/health` → `{"status":"ok","model_loaded":true,"vector_store_ready":true}`
(`vector_store_ready` is `false` for the first ~20s while sentence-transformers loads in a background thread; the app is usable immediately.)

### Switching the classifier

```bash
AROGYA_MODEL=multinomial  # or bernoulli (default)
```

Both variants are trained and saved by `train_model.py`.

---

## Evaluation

Every number below was produced by a script in `eval/` and is served live at `/api/evaluation` and rendered in the in-app **Evaluation** dashboard. Nothing is hand-copied.

```bash
python eval/compare_models.py   # -> eval/results/model_comparison.json
python eval/eval_nlp.py         # -> eval/results/nlp_eval.json
python eval/eval_rag.py         # -> eval/results/rag_eval.json
```

### Classifier — 8 models, identical splits and 5-fold CV

| Model | Accuracy | Top-3 | Macro F1 | CV accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.851 | 0.963 | **0.826** | 0.858 ± 0.014 |
| **Bernoulli NB** *(deployed)* | 0.838 | 0.958 | 0.792 | 0.852 ± 0.016 |
| Random Forest | 0.816 | 0.952 | 0.779 | 0.821 ± 0.016 |
| Linear SVM | 0.803 | 0.930 | 0.769 | 0.818 ± 0.018 |
| Multinomial NB | 0.811 | 0.950 | 0.762 | 0.828 ± 0.014 |
| k-NN (k=5) | 0.796 | 0.913 | 0.721 | 0.803 ± 0.006 |
| Decision Tree | 0.706 | 0.723 | 0.631 | 0.688 ± 0.021 |
| Majority baseline | 0.050 | 0.105 | 0.002 | 0.050 ± 0.001 |

**Why Bernoulli NB is deployed despite Logistic Regression scoring 0.033 higher macro F1:**

1. Its decision **decomposes exactly** into per-symptom log-odds, which is what makes the explanation panel a true attribution rather than an approximation. `tests/test_explain.py` asserts the reported weights reconstruct sklearn's own joint log-likelihood.
2. Unlike Multinomial NB it scores symptom **absence**, so the NegEx layer's output — symptoms the user explicitly denied — becomes usable evidence instead of just being omitted.
3. It fits in ~18 ms, keeping the whole pipeline reproducible on a laptop.

### NLP components — vs a hand-annotated gold standard

| Component | Metric | Score |
|---|---|---|
| Symptom extraction | micro F1 / macro F1 | **0.980 / 0.983** |
| | exact set match | 96.7% |
| | **negation leaks** | **0** |
| Assertion (NegEx/ConText) | accuracy over 6 labels | **100%** (30 mentions) |
| Temporal | value / bucket accuracy | **100% / 100%** |
| Trend detection | accuracy | 100% |
| Distress detection | accuracy | 100% |
| Intent | cross-validated accuracy | 0.758 |
| | held-out hard paraphrases | 0.656 |

"Negation leaks" counts symptoms the user explicitly denied that still reached the classifier — the failure mode that would actually mislead someone.

The two intent numbers are both reported deliberately. The held-out set is 32 paraphrases chosen to be hard ("much obliged", "assalam walekum") and sharing **no sentence** with the 360-example corpus; the evaluation script asserts zero overlap. Because intent misclassification is cheap to recover from but expensive to act on, the dialogue policy requires **45% confidence** before routing on an intent — every miss in the held-out set fell below that threshold and was safely handled as an ordinary symptom description.

### Retrieval

| Mode | Hit@1 | Hit@3 | MRR | nDCG@5 |
|---|---|---|---|---|
| Unfiltered semantic search | 0.844 | 0.906 | 0.878 | 0.872 |
| Filtered by prediction *(production path)* | 1.000 | 1.000 | 1.000 | 1.000 |

The filtered row is near-perfect by construction and is **not** a measure of retrieval quality — it is reported separately for exactly that reason. Precision@5 is 0.331 against a hard ceiling of 0.394 (each disease document yields only ~2 chunks, so at most 2 of 5 results can be relevant) — i.e. 84% of what is achievable.

### Dataset

The synthetic generator assigns each symptom a **per-disease prevalence** rather than marking every core symptom present in every row, and class sizes follow coarse **clinical base rates** rather than being uniform. Both changes matter:

- with 100%-prevalent core symptoms the data was trivially separable, every model scored ~100%, and all per-symptom log-likelihoods were identical, so the explanation could not say which symptom mattered;
- with uniform class sizes the Naive Bayes prior carried no information, and "fever and body aches for 3 days" came back as **AIDS at 85% confidence**. With base rates encoded it returns Dengue 46% / Typhoid 35%, and AIDS still surfaces correctly for its own presentation.

---

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/session` | POST | Create a new anonymous chat session |
| `/api/chat` | POST | Send a message; returns reply, state, NLP trace, prediction |
| `/api/chat/history/{id}` | GET | Full message + state history |
| `/api/health` | GET | Backend + model + vector store status |
| **`/api/nlp/analyze`** | POST | **Full linguistic analysis of any sentence** — tokens, stems, entities with spans, assertion cues and scope, temporal values, intent posterior, distress, per-stage timings |
| **`/api/nlp/capabilities`** | GET | Pipeline stages and the size of every lexical resource |
| **`/api/nlp/explain`** | POST | Per-symptom Naive Bayes attribution for an arbitrary symptom set |
| **`/api/evaluation`** | GET | Every measured result, straight from `eval/results/` |
| `/api/debug/predict` | POST | Raw symptom list → top predictions |
| `/api/session/{id}/report` | GET | Consultation summary (JSON + printable markdown) |
| `/api/stats` | GET | Live model/corpus facts |

---

## Safety layer

On **every** user turn — before any LLM call, prediction, or Ayurvedic output — the red-flag layer scans the message and accumulated symptoms for cardiac, stroke, respiratory, bleeding, anaphylaxis, obstetric, abdominal and mental-health crisis patterns. On a match the conversation jumps to a terminal `emergency` state: no diagnosis is given, only "seek immediate care" guidance with local emergency numbers (112 / 108 in India).

`ML_SYMPTOM_ALIASES` maps the ML feature vocabulary onto the rule vocabulary (`breathlessness` → `shortness_of_breath`), so symptoms accumulated across several turns still reach the rules.

## State machine

```
GREETING → SYMPTOM_COLLECTION ──(red-flag check every turn)──► EMERGENCY (terminal)
   SYMPTOM_COLLECTION → CLARIFYING (loop, 3-6 questions) → RESULT
   RESULT → question about the result   → answered from the stored assessment
   RESULT → new symptom description     → fresh consultation
```

State, accumulated symptoms, denied symptoms and the last prediction persist on the SQLite session row, so a page refresh restores the conversation and follow-up questions are answered from the assessment the user is actually looking at.

## Tests

```bash
python -m pytest tests/ -q      # 142 tests
```

Covers the Porter stemmer against its published reference vocabulary, negation scope and termination, code-mixed normalization, offset round-tripping, the three ConText axes, temporal normalization, distress banding, and — most importantly — that the explanation layer's reported weights reconstruct the estimator's own joint log-likelihood exactly.

## License / data notes

- The dataset is synthetic (`ml/generate_data.py`), structured after well-known public symptom–disease datasets. To use the original Kaggle CSV, place it in `ml/data/` before running `train_model.py`.
- Knowledge base content in `rag/knowledge_base/` is original general public-health writing, not scraped. No drug names or dosages are recommended anywhere.
