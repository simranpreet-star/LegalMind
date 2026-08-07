"""
LegalMind — Contract Clause Analyser
Streamlit entry point.

Run locally:  streamlit run app.py
Deployed at:  https://legalmind-clauses.streamlit.app
"""

from __future__ import annotations

import os
import sys

# src/ on the path so all modules resolve without installation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st

st.set_page_config(
    page_title="LegalMind · Contract Clause Analyser",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── imports after st.set_page_config ──────────────────────────────────────────
import json
import tempfile

from classical_models import load_model, predict_with_abstention, MODELS_WITH_PROBA
from data_loader import load_label_mapping
from pdf_parser import parse_contract_pdf, parse_contract_text
from risk_assessment import assess_document_risk, RISK_LEVELS
from rewrite_suggestions import generate_rewrite

# ── paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LABEL_MAPPING_PATH = os.path.join(MODELS_DIR, "label_mapping.json")
MODEL_META_PATH = os.path.join(MODELS_DIR, "model_meta.json")

# Models that expose predict_proba and can therefore use the abstention threshold
_DEPLOYABLE = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest":       "random_forest.joblib",
    "XGBoost":             "xgboost.joblib",
}

RISK_COLORS = {
    "High":       "#e74c3c",
    "Medium":     "#f39c12",
    "Low":        "#27ae60",
    "Unassessed": "#95a5a6",
}

# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def _load(model_filename: str):
    mapping = load_label_mapping(LABEL_MAPPING_PATH)
    model   = load_model(os.path.join(MODELS_DIR, model_filename))
    return model, mapping


def _available_models():
    if not os.path.isdir(MODELS_DIR):
        return {}
    return {
        name: fname
        for name, fname in _DEPLOYABLE.items()
        if os.path.exists(os.path.join(MODELS_DIR, fname))
    }


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scales.png", width=64)
    st.title("LegalMind")
    st.caption("Contract clause analysis · v2.0")
    st.divider()

    avail = _available_models()
    if not avail:
        st.error("No trained models found in models/. Run `python scripts/train_classical.py`.")
        st.stop()

    # show macro F1 next to each model name if metadata is available
    meta = {}
    if os.path.exists(MODEL_META_PATH):
        with open(MODEL_META_PATH) as f:
            meta = json.load(f)

    model_labels = {
        name: f"{name}  (F1 {meta[name]['macro_f1']:.3f})" if name in meta else name
        for name in avail
    }
    chosen_label = st.selectbox("Classification model", list(model_labels.values()))
    model_name   = [k for k, v in model_labels.items() if v == chosen_label][0]

    confidence_threshold = st.slider(
        "Abstention threshold",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Clauses where the model's confidence is below this are marked "
             "Unclassified rather than being forced into a category."
    )

    do_rewrites = st.checkbox(
        "Generate rewrite suggestions",
        value=False,
        help="Calls Groq or Anthropic (if API key is set) to draft lower-risk alternatives "
             "for Medium/High clauses. Uses a template fallback without an API key.",
    )

    st.divider()
    st.caption("Target categories: Governing Law · Termination · Liability · Payment · Exclusivity")
    st.caption("Clauses outside these five categories are marked Unclassified.")

# ── main panel ─────────────────────────────────────────────────────────────────
st.title("⚖️ LegalMind Contract Clause Analyser")
st.write(
    "Upload a contract PDF **or** paste clause text below. "
    "The model will classify each clause, score its risk, and optionally suggest a rewrite."
)

tab_upload, tab_paste = st.tabs(["📄 Upload PDF", "📋 Paste text"])

clauses = []

with tab_upload:
    uploaded = st.file_uploader("Upload a contract PDF", type=["pdf"])
    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        with st.spinner("Extracting clauses …"):
            try:
                clauses = parse_contract_pdf(tmp_path)
                st.success(f"Extracted {len(clauses)} candidate clauses from the PDF.")
            except Exception as exc:
                st.error(f"PDF extraction failed: {exc}")
        os.unlink(tmp_path)

with tab_paste:
    pasted = st.text_area(
        "Paste contract text here",
        height=200,
        placeholder="Paste one or more clauses …",
    )
    if st.button("Analyse pasted text") and pasted.strip():
        clauses = parse_contract_text(pasted)
        if not clauses:
            clauses = [{"header": None, "text": pasted.strip()}]
        st.success(f"Found {len(clauses)} candidate clauses.")

# ── analysis ───────────────────────────────────────────────────────────────────
if clauses:
    with st.spinner("Classifying …"):
        model, mapping = _load(_DEPLOYABLE[model_name])
        id2label = mapping["id2label"]
        texts = [c["text"] for c in clauses]

        preds, confidences = predict_with_abstention(
            model, texts, threshold=confidence_threshold
        )

        classified = [
            {
                "text":       clause["text"],
                "header":     clause.get("header"),
                "category":   "Unclassified" if pred == -1 else id2label[int(pred)],
                "confidence": float(conf),
            }
            for clause, pred, conf in zip(clauses, preds, confidences)
        ]

    risk_result = assess_document_risk(classified)
    cov = risk_result["coverage"]
    summary = risk_result["summary"]

    # ── risk summary strip ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Risk Summary")

    cols = st.columns(5)
    for col, level in zip(cols, ["High", "Medium", "Low", "Unassessed"]):
        col.metric(level, summary[level],
                   delta=None,
                   help=f"Number of clauses rated {level}")
    cols[4].metric(
        "Rule coverage",
        f"{cov['pct_covered']:.0f}%",
        help=f"Share of clauses where at least one risk pattern matched "
             f"({cov['n_rules_fired']}/{cov['n_clauses']}). "
             "Unassessed does not mean low risk — the rulebook may simply not cover "
             "this clause type."
    )

    # ── per-clause detail ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Clause Detail")

    filter_options = ["All"] + [lvl for lvl in ["High", "Medium", "Low", "Unassessed"]
                                 if summary[lvl] > 0]
    risk_filter = st.selectbox("Show risk level", filter_options, index=0)

    shown = 0
    for c in risk_result["clauses"]:
        if risk_filter != "All" and c["risk"] != risk_filter:
            continue
        shown += 1

        color = RISK_COLORS.get(c["risk"], "#bdc3c7")
        cat_label = c["category"]
        risk_label = c["risk"]
        conf_pct = f"{c['confidence']:.0%}"

        header_txt = (
            f"**{c['header']}** — " if c.get("header") else ""
        )
        expander_label = (
            f"{header_txt}[{cat_label}] :{risk_label}: · confidence {conf_pct}"
        )

        with st.expander(expander_label):
            st.markdown(
                f"<div style='border-left:4px solid {color}; padding:8px 12px; "
                f"background:#f9f9f9; border-radius:4px; font-size:0.93em'>"
                f"{c['text']}</div>",
                unsafe_allow_html=True,
            )
            if c.get("matched_patterns"):
                st.caption("⚑ Flagged patterns: " + " · ".join(c["matched_patterns"]))
            elif c["risk"] == "Unassessed":
                st.caption("ℹ No pattern matched — clause is unassessed, not confirmed low risk.")

            if do_rewrites and c["risk"] in ("Medium", "High"):
                with st.spinner("Generating rewrite suggestion …"):
                    rw = generate_rewrite(
                        c["text"], c["category"], c["risk"], c.get("matched_patterns", [])
                    )
                if rw:
                    source_badge = {
                        "groq":              "🤖 Groq",
                        "anthropic":         "🤖 Anthropic",
                        "template-fallback": "📝 Template (no API key)",
                    }.get(rw["source"], rw["source"])

                    st.markdown(f"**Suggested rewrite** · {source_badge}")
                    st.info(rw["rewrite"])
                    if rw["changes"]:
                        st.markdown("**Changes made:**")
                        for ch in rw["changes"]:
                            st.write(f"- {ch}")

    if shown == 0:
        st.info("No clauses match the current filter.")

else:
    st.info("Upload a PDF or paste contract text above to begin.")

# ── footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ LegalMind is a research prototype. Risk ratings are based on keyword heuristics "
    "and classifier predictions — they are not legal advice. Always consult a qualified "
    "lawyer before acting on any analysis. "
    "· [GitHub](https://github.com/simranpreet-star/LegalMind)"
)
