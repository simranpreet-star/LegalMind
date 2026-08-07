"""
Train all four classical models on the five-category CUAD dataset and save
the fitted pipelines to models/.

Each model is an imblearn Pipeline (TF-IDF → [resampler] → classifier) so
the vectoriser is always fitted inside each CV fold and never on evaluation
data. Hyperparameter selection uses document-grouped GridSearchCV so the
inner folds are also contract-disjoint.

Usage:
    python scripts/train_classical.py            # full grid search
    python scripts/train_classical.py --no-tune  # skip search, fixed params
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import paths
paths.ensure_dirs()

from data_loader import get_five_category_dataset
from splits import grouped_split, measure_text_overlap, describe_split
from classical_models import TRAINERS, timed_predict, save_model
from evaluation import evaluate_predictions, plot_confusion_matrix, save_comparison_table, save_per_class_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true", help="skip grid search")
    ap.add_argument("--models", nargs="+", default=list(TRAINERS),
                    choices=list(TRAINERS), help="which models to train")
    args = ap.parse_args()
    tune = not args.no_tune

    print("loading CUAD dataset …")
    df, mapping = get_five_category_dataset()
    print(f"  {len(df):,} clause spans | {df['doc_id'].nunique()} source contracts "
          f"| {df['label'].nunique()} categories")
    print(f"  label distribution:\n{df['label'].value_counts().to_string()}")

    print("\nsplitting by source contract (doc_id) …")
    train_df, val_df, test_df = grouped_split(df)
    for row in describe_split(train_df, val_df, test_df):
        print(f"  {row['split']:5s}  {row['rows']:5d} rows | {row['contracts']:3d} contracts")

    overlap = measure_text_overlap(train_df, test_df)
    print(f"  exact-duplicate overlap train↔test: "
          f"{overlap['n_overlapping_unique_clauses']} clauses "
          f"({overlap['pct_of_test_rows']:.1f}% of test rows)")

    # train on full train set; val is kept for threshold tuning if needed
    train_texts = train_df["text"].tolist()
    test_texts  = test_df["text"].tolist()
    y_train = train_df["label_id"].values
    y_test  = test_df["label_id"].values
    groups  = train_df["doc_id"].values
    id2label = mapping["id2label"]

    results = []
    for name in args.models:
        print(f"\n{'─'*60}\ntraining {name}"
              + (" (grid search, grouped CV)" if tune else " (fixed params)") + " …")
        trainer_fn = TRAINERS[name]
        model, best_params = trainer_fn(train_texts, y_train, groups, tune=tune)
        if best_params:
            print(f"  best params: {best_params}")

        preds, elapsed = timed_predict(model, test_texts)
        result = evaluate_predictions(y_test, preds, id2label, name,
                                      inference_time_sec=elapsed)
        results.append(result)

        slug = name.lower().replace(" ", "_")
        plot_confusion_matrix(y_test, preds, id2label, name, paths.MODELS_DIR)
        save_model(model, os.path.join(paths.MODELS_DIR, f"{slug}.joblib"))
        print(f"  saved → models/{slug}.joblib")

    # label mapping is already written to MODELS_DIR by build_and_save_label_mapping

    save_comparison_table(results, os.path.join(paths.MODELS_DIR, "classical_comparison.json"))
    save_per_class_table(results, os.path.join(paths.MODELS_DIR, "per_class_f1.csv"))

    # write a small metadata file the app reads to know which models exist
    meta = {m["model"]: {"macro_f1": m["macro_f1"]} for m in results}
    with open(os.path.join(paths.MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\ndone — models saved to", paths.MODELS_DIR)


if __name__ == "__main__":
    main()
