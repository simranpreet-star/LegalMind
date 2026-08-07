"""
Document-grouped train/val/test splitting and cross-validation.

Why not a plain stratified split: CUAD annotates the same span against
several questions, and standard contract boilerplate ("This Agreement shall
be governed by the laws of...") recurs across many contracts. A row-level
random split therefore leaves near-duplicate text on both sides of the
boundary and the reported score measures memorisation as much as
generalisation.

Everything here splits on `doc_id`, so a source contract lands wholly in one
partition. `measure_text_overlap` is kept as a standing check -- run it after
any split and it should return ~0.
"""

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

RANDOM_STATE = 42


def grouped_split(df, test_size=0.1, val_size=0.1, random_state=RANDOM_STATE, group_col="doc_id"):
    """
    80/10/10 train/val/test, split so that no source contract appears in more
    than one partition.

    Note the class distribution is not controlled: GroupShuffleSplit cannot
    stratify and hold groups intact at the same time. Check the resulting
    per-split label counts rather than assuming they match.
    """
    gss = GroupShuffleSplit(
        n_splits=1, test_size=(test_size + val_size), random_state=random_state
    )
    train_idx, temp_idx = next(gss.split(df, groups=df[group_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    relative_test = test_size / (test_size + val_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=random_state)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))

    return (
        train_df,
        temp_df.iloc[val_idx].reset_index(drop=True),
        temp_df.iloc[test_idx].reset_index(drop=True),
    )


def measure_text_overlap(train_df, test_df, text_col="text"):
    """
    Exact-duplicate clause overlap between two partitions, as a count and as
    a share of the second partition. A grouped split should give ~0; a
    row-level stratified split on CUAD does not.
    """
    train_texts = set(train_df[text_col].str.strip().str.lower())
    test_texts = set(test_df[text_col].str.strip().str.lower())
    overlap = train_texts & test_texts
    return {
        "n_overlapping_unique_clauses": len(overlap),
        "pct_of_test_rows": float(
            test_df[text_col].str.strip().str.lower().isin(overlap).mean() * 100
        ),
    }


def grouped_cv_indices(df, n_splits=5, group_col="doc_id"):
    """
    Document-grouped CV folds, yielded as (train_idx, val_idx) positional
    arrays. Used both for reporting cross-validated scores and as the `cv`
    argument to GridSearchCV, so hyperparameter selection is subject to the
    same document-disjointness as the final evaluation.
    """
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=df[group_col]))


def describe_split(train_df, val_df, test_df, label_col="label"):
    """Row counts, contract counts and per-class breakdown for each partition."""
    rows = []
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        counts = part[label_col].value_counts().to_dict()
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "contracts": part["doc_id"].nunique(),
                **{f"n_{k}": v for k, v in sorted(counts.items())},
            }
        )
    return rows


def class_weight_array(y, n_classes):
    """
    Balanced class weights normalised to average 1.0, for the transformer
    loss function. Kept here so the classical and transformer paths derive
    weights from the same definition.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return weights
