# LegalMind — Contract Clause Analyser

**Live app → https://legalmind-clauses.streamlit.app**

Upload a contract PDF or paste clause text. LegalMind classifies each clause into one of five categories, scores its legal risk, and optionally suggests a lower-risk rewrite.

---

## What it does

| Stage | Description |
|-------|-------------|
| **Parse** | Numbered-header segmentation of uploaded PDFs; regex sentence fallback for unstructured text |
| **Classify** | TF-IDF + classical ML (Logistic Regression / Random Forest / XGBoost), trained on [CUAD](https://www.atticusprojectai.org/cuad) with document-grouped train/test splits |
| **Risk triage** | Rule-based keyword matching → High / Medium / Low / Unassessed per clause |
| **Rewrite** | Groq or Anthropic LLM suggestions for risky clauses; template fallback without an API key |

**Target clause categories:** Governing Law · Termination · Liability · Payment · Exclusivity
Clauses outside these five are marked *Unclassified* via a calibrated abstention threshold.

---

## Repository layout

```
app.py                  Streamlit entry point
requirements.txt
src/
  paths.py              Path config (auto-detects Kaggle / local / Streamlit Cloud)
  data_loader.py        CUAD loading and 5-category label mapping
  splits.py             Document-grouped train/val/test splits and k-fold CV
  classical_models.py   Pipeline-based LR / SVM / RF / XGBoost with leak-free CV
  evaluation.py         Shared metrics, confusion matrices, comparison tables
  pdf_parser.py         PDF and plain-text clause segmentation
  risk_assessment.py    Rule-based risk scoring (Unassessed != Low)
  rewrite_suggestions.py Groq -> Anthropic -> template rewrite chain
  rag_pipeline.py       Dual-corpus retrieval (CUAD + ILDC) for RAG context
scripts/
  train_classical.py    End-to-end training script
models/                 Committed trained artifacts (.joblib + label_mapping.json)
```

---

## Running locally

```bash
git clone https://github.com/simranpreet-star/LegalMind.git
cd LegalMind
pip install -r requirements.txt
streamlit run app.py
```

The trained models are committed in `models/` so the app works out of the box without retraining.

### Retraining from CUAD

Download `CUAD_v1.json` from https://www.atticusprojectai.org/cuad and place it in `data/`:

```bash
python scripts/train_classical.py            # grid search (slow, accurate)
python scripts/train_classical.py --no-tune  # fixed params (fast, smoke test)
```

### Optional: LLM rewrites

Set one of these environment variables (or add to `.streamlit/secrets.toml`):
```
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Key methodological decisions

**Document-grouped splits** — CUAD annotates the same text span against multiple questions and boilerplate recurs across contracts. A plain row-level stratified split produces near-duplicate clauses on both sides of the train/test boundary and inflates reported F1. All splits here are by `doc_id` (source contract title).

**Leak-free grid search** — Every model is an `imblearn.Pipeline` that includes the TF-IDF vectoriser. The pipeline is what gets passed to `GridSearchCV`, so the vectoriser is fitted inside each fold, not on the full training set. The CV uses `GroupKFold` on `doc_id`.

**Unassessed != Low risk** — When no risk pattern matches a clause, it is reported as *Unassessed*, not *Medium*. Collapsing those two makes the risk distribution uninterpretable. The app shows rule coverage so you know how much of the document the rulebook actually reached.

**Abstention threshold** — The five target classes cover ~12 of CUAD's 41 categories. Most clauses in a real contract belong to none of them. The confidence threshold (default 0.6) lets the classifier abstain rather than force an incorrect label.

---

## Limitations

- Risk scores are keyword heuristics, not legal judgments. Rule coverage on a given document may be low.
- Training data is US commercial contracts (CUAD). Performance on Indian or other jurisdictions is lower.
- Governing Law clauses score near-perfectly because they are a single homogeneous CUAD category; this inflates the macro F1 average.
- This is a research prototype. Do not rely on it as legal advice.

---

## License

MIT — see [LICENSE](LICENSE).
CUAD dataset: CC BY 4.0, © The Atticus Project.
