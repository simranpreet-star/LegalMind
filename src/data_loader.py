"""
CUAD loading and label mapping.

Every row carries the source contract title as `doc_id`. This matters: CUAD
annotates the same span against multiple questions, and contract boilerplate
repeats across documents in the corpus, so a random row-level split puts
near-identical text on both sides of the train/test boundary. All splitting
in this project is document-grouped (see splits.py) and that is only possible
because doc_id is preserved here.

Label mappings are derived from the filtered dataframe at runtime
(sorted(df['label'].unique())) and persisted to label_mapping.json rather
than hardcoded, so the ordering can't silently drift between the training
run and the app that loads the artifacts.
"""

import json
import os

import pandas as pd

from paths import CUAD_JSON_PATH, LABEL_MAPPING_PATH

# CUAD has 41 raw categories; this project targets five. The groups below are
# the exact CUAD category names that roll up into each target class.
#
# Two caveats worth carrying into any writeup:
#   - CUAD has no "Confidentiality" category. Exclusivity (grouped with
#     Non-Compete / No-Solicit) is used as an imperfect proxy.
#   - The groups are very uneven in how much internal variety they contain.
#     "Governing Law" is a single category of near-identical boilerplate and
#     is close to trivially separable, while "Payment" and "Exclusivity" each
#     merge four categories. Report per-class F1, not just macro, or the easy
#     class will flatter the average.
CUAD_CATEGORY_GROUPS = {
    "Governing Law": [
        "Governing Law",
    ],
    "Termination": [
        "Termination For Convenience",
    ],
    "Liability": [
        "Cap On Liability",
        "Uncapped Liability",
    ],
    "Payment": [
        "Price Restrictions",
        "Minimum Commitment",
        "Volume Restriction",
        "Revenue/Profit Sharing",
    ],
    "Exclusivity": [
        "Exclusivity",
        "Non-Compete",
        "No-Solicit Of Customers",
        "No-Solicit Of Employees",
    ],
}

TARGET_CATEGORIES = sorted(CUAD_CATEGORY_GROUPS)


def load_cuad_from_local_json(path=None):
    path = path or CUAD_JSON_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CUAD_v1.json not found at {path}. Download it from "
            "https://www.atticusprojectai.org/cuad and place it in data/, or "
            "point the CUAD_JSON_PATH environment variable at it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_category_from_question(question):
    """CUAD phrases each question as: ... related to "Category Name" ..."""
    if '"' in question:
        parts = question.split('"')
        if len(parts) >= 2:
            return parts[1]
    return None


def cuad_json_to_dataframe(raw_cuad):
    """
    Flattens CUAD's SQuAD-style nesting into one row per annotated span,
    keeping the source contract title so splits can be document-grouped.
    """
    rows = []
    for doc_idx, doc in enumerate(raw_cuad.get("data", [])):
        doc_title = doc.get("title", f"doc_{doc_idx}")
        for para in doc.get("paragraphs", []):
            for qa in para.get("qas", []):
                category = _extract_category_from_question(qa.get("question", ""))
                if not category:
                    continue
                for ans in qa.get("answers", []):
                    span_text = ans.get("text", "")
                    if span_text:
                        rows.append(
                            {"text": span_text, "raw_label": category, "doc_id": doc_title}
                        )
    return pd.DataFrame(rows)


def map_to_five_categories(df, keep_unmapped=False):
    """
    Rolls CUAD's 41 categories up into the five target classes.

    keep_unmapped=True instead returns the rows that fall *outside* the five
    classes. Those are the out-of-scope set used by
    scripts/evaluate_out_of_scope.py to measure how often the confidence
    threshold correctly declines to guess -- which is most of what a real
    uploaded contract actually contains.
    """
    reverse_map = {
        src: target for target, sources in CUAD_CATEGORY_GROUPS.items() for src in sources
    }

    df = df.copy()
    df["label"] = df["raw_label"].map(reverse_map)

    if keep_unmapped:
        return df[df["label"].isna()].drop(columns=["label"]).reset_index(drop=True)
    return df.dropna(subset=["label"]).reset_index(drop=True)


def build_and_save_label_mapping(df, out_path=None):
    out_path = out_path or LABEL_MAPPING_PATH
    labels = sorted(df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    mapping = {"label2id": label2id, "id2label": id2label}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    return mapping


def load_label_mapping(path=None):
    path = path or LABEL_MAPPING_PATH
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    mapping["id2label"] = {int(k): v for k, v in mapping["id2label"].items()}
    return mapping


def get_five_category_dataset(cuad_json_path=None, label_mapping_path=None):
    """Returns (df with text/label/label_id/doc_id, mapping)."""
    raw = load_cuad_from_local_json(cuad_json_path)
    df = map_to_five_categories(cuad_json_to_dataframe(raw))
    mapping = build_and_save_label_mapping(df, label_mapping_path)
    df["label_id"] = df["label"].map(mapping["label2id"])
    return df, mapping


def get_out_of_scope_dataset(cuad_json_path=None):
    """
    CUAD spans whose category is none of the five target classes. A deployed
    classifier will meet these constantly and has no correct label available
    for them, so the only right answer is to abstain.
    """
    raw = load_cuad_from_local_json(cuad_json_path)
    return map_to_five_categories(cuad_json_to_dataframe(raw), keep_unmapped=True)
