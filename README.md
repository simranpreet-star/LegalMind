# LegalMind — Contract Clause Analyser

**Live app → https://legalmind-clauses.streamlit.app**

Upload a contract PDF or paste clause text. LegalMind splits it into clauses, classifies each clause into one of five categories, flags risky language with a rule-based checker, and can optionally suggest a lower-risk rewrite using an LLM.

---

## What the app does

| Stage | Description |
|-------|-------------|
| **Parse** | Splits PDFs (via `pdfplumber`) or pasted text on numbered clause headers. Headers are lines like `1.`, `2.3`, `ARTICLE IV` or `SECTION 5`. If fewer than 2 are found, falls back to a regex sentence splitter that groups sentences into chunks of up to 4 (chunks under 20 characters are dropped). Pasted text that yields no chunks is treated as one clause. |
| **Classify** | TF-IDF (1–2 grams) + classical ML. The app lets you pick **Logistic Regression**, **Random Forest** or **XGBoost**. Trained on [CUAD](https://www.atticusprojectai.org/cuad) with document-grouped splits. Class imbalance is handled with balanced class weights (LR, RF) or random oversampling (XGBoost). The picker shows each model's macro F1 from `models/model_meta.json`. |
| **Abstain** | If the model's top-class probability is below a confidence threshold (default **0.6**, adjustable with a slider in the app), the clause is marked **Unclassified** instead of being forced into a category. |
| **Risk triage** | Regex rules specific to the **predicted category** (7–13 patterns per category) → **High / Medium / Low** by the most severe level matched, **Unassessed** when none match. Clauses marked **Unclassified** have no rules, so they are always Unassessed. The app shows **rule coverage** (share of clauses any rule matched). |
| **Rewrite** *(optional)* | For Medium/High-risk clauses only, when the checkbox is enabled. Tries Groq (`llama-3.3-70b-versatile`), then Anthropic (`claude-sonnet-4-6`), then a template-based phrase-softening fallback if no key is set or both providers fail. Each suggestion is labelled with the source that produced it. |

**Target categories:** Governing Law · Termination · Liability · Payment · Exclusivity

These five classes are built by grouping 12 of CUAD's 41 raw categories:

| Target class | CUAD categories |
|---|---|
| Governing Law | Governing Law |
| Termination | Termination For Convenience |
| Liability | Cap On Liability, Uncapped Liability |
| Payment | Price Restrictions, Minimum Commitment, Volume Restriction, Revenue/Profit Sharing |
| Exclusivity | Exclusivity, Non-Compete, No-Solicit Of Customers, No-Solicit Of Employees |

CUAD has no Confidentiality category, so Exclusivity is used as an imperfect proxy for restrictive-covenant clauses.

---

## Results

Held-out test set: **363 clauses** (≈10% split), document-grouped (no contract appears in both train and test).

These metrics come from the model's plain `predict()` output, **with no abstention threshold applied**. In the app, low-confidence clauses become Unclassified, so what you see there will differ from these numbers.

**Macro F1 by model** (as committed in `models/model_meta.json`):

| Model | Macro F1 |
|---|---|
| Logistic Regression | **0.928** |
| Random Forest | 0.900 |
| XGBoost | 0.899 |

**Per-class results — Logistic Regression** (derived from `models/confusion_matrix_logistic_regression.png`):

| Class | Test clauses | Precision | Recall | F1 |
|---|---|---|---|---|
| Exclusivity | 106 | 0.93 | 0.98 | 0.95 |
| Governing Law | 50 | 1.00 | 0.98 | 0.99 |
| Liability | 80 | 0.97 | 0.93 | 0.95 |
| Payment | 100 | 0.94 | 0.91 | 0.92 |
| Termination | 27 | 0.79 | 0.85 | **0.82** |

How to read this:
- Governing Law is near-perfect because it is a single category of near-identical boilerplate; it lifts the macro average.
- Termination is the weakest class and has the smallest test support (27), so its score is also the least stable.
- These numbers are **in-scope only** — every test clause belongs to one of the five classes. They do not measure how well the abstention threshold rejects out-of-scope clauses, which make up most of a real contract (see *Limitations*).

Confusion matrices for all trained models are in `models/`.

---

## Repository layout

```
app.py                   Streamlit entry point
requirements.txt
src/
  paths.py               Path config: detects Kaggle; otherwise uses repo-relative paths
                         (local and Streamlit Cloud). Overridable via env vars.
  data_loader.py         CUAD loading and 41 → 5 category mapping
  splits.py              Document-grouped train/val/test splits, grouped k-fold, overlap check
  classical_models.py    Pipeline-based LR / SVM / RF / XGBoost, grouped grid search,
                         abstention prediction
  evaluation.py          Metrics, confusion matrices, comparison tables
  pdf_parser.py          PDF and plain-text clause segmentation
  risk_assessment.py     Rule-based risk scoring and rule coverage
  rewrite_suggestions.py Groq → Anthropic → template rewrite chain
  rag_pipeline.py        Experimental dual-corpus retrieval (CUAD + ILDC) — not used by the app
scripts/
  train_classical.py     End-to-end training script
models/                  Committed trained models (.joblib), label mapping, metrics, confusion matrices
```

**About SVM:** `train_classical.py` also trains a `LinearSVC` and `models/svm.joblib` is committed, but the app does not offer it. `LinearSVC` has no `predict_proba`, so it cannot use the abstention threshold.

---

## Running locally

```bash
git clone https://github.com/simranpreet-star/LegalMind.git
cd LegalMind
pip install -r requirements.txt
streamlit run app.py
```

The trained models are committed in `models/`, so the app works without retraining.

### Optional: LLM rewrites

Set one of these as environment variables, or add them to `.streamlit/secrets.toml` (git-ignored):

```
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
```

Without a key, rewrites use the template fallback, which swaps a small set of high-risk phrases (e.g. "sole discretion", "without cause", "unlimited liability") for softer wording.

### Retraining from CUAD

Download `CUAD_v1.json` from https://www.atticusprojectai.org/cuad and place it in `data/` (or set `CUAD_JSON_PATH`):

```bash
python scripts/train_classical.py            # grid search (slow)
python scripts/train_classical.py --no-tune  # fixed params (fast smoke test)
```

The script trains all four models and writes to `models/`: the `.joblib` models, `label_mapping.json`, `model_meta.json`, confusion matrices, `classical_comparison.json` and `per_class_f1.csv`. (The last two are not currently committed.) A default run writes an SVM entry to `model_meta.json` too; the app ignores it because SVM isn't offered. The validation split is created but not currently used.

---

## Experimental: dual-corpus retrieval (`src/rag_pipeline.py`)

A retrieval module built as groundwork for retrieval-augmented rewrites. **It is not yet connected to the app** — rewrites currently use only the clause text, category, risk level and flagged phrases.

What it contains:
- **Embeddings:** SentenceTransformer `all-MiniLM-L6-v2`.
- **Two collections:** CUAD clause spans (with category labels), and contract-relevant excerpts from ILDC Indian court judgments (keyword-filtered chunks of up to 800 characters).
- **Vector store:** ChromaDB when available; otherwise an in-memory numpy/cosine-similarity fallback with the same interface.
- **Retrieval + context:** top-k from each corpus, formatted into a prompt-ready context block.
- **Evaluation:** Recall@k and MRR (label-based relevance), compared against a majority-label baseline.

Its dependencies are **not** in `requirements.txt`. To experiment with it:

```bash
pip install sentence-transformers chromadb
```

---

## Key methodological decisions

**Document-grouped splits** — CUAD annotates the same text span against multiple questions, and boilerplate recurs across contracts. A row-level stratified split puts near-duplicate clauses on both sides of the train/test boundary and inflates F1. All splits here group by `doc_id` (source contract title). Trade-off: `GroupShuffleSplit` cannot stratify, so class balance across splits is not controlled.

**Leak-free grid search** — Every model is an `imblearn.Pipeline` that includes the TF-IDF vectoriser, and the pipeline is what goes into `GridSearchCV`. The vectoriser is therefore fitted inside each fold, and CV uses `GroupKFold` on `doc_id`.

**Unassessed ≠ Low ≠ Medium** — When no risk rule matches a clause, it is reported as *Unassessed* rather than defaulting to any risk level. Otherwise, the risk distribution can't be interpreted. Rule coverage is shown so you can see how much of the document the rulebook actually reached.

**Confidence-threshold abstention** — The five target classes cover 12 of CUAD's 41 categories, so most clauses in a real contract belong to none of them. A fixed probability threshold (default 0.6) lets the classifier abstain rather than force a label. The threshold is a manual default — it is not calibrated or tuned against out-of-scope data, and the models' probabilities are not calibrated (Random Forest and XGBoost probabilities in particular tend to be poorly calibrated).

---

## Limitations

- **Abstention is not evaluated.** Reported metrics cover in-scope clauses only. How often the threshold correctly rejects out-of-scope clauses has not been measured yet.
- **Risk scores are keyword heuristics, not legal judgments.** Rule coverage on a given document may be low.
- **Risk depends on the classifier.** Rules are chosen by predicted category, so a misclassified clause is checked against the wrong rules, and Unclassified clauses are never risk-checked or rewritten.
- **Small test set.** 363 test clauses in total, and only 27 for Termination.
-
