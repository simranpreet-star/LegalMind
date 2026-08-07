"""
Shared metrics and plots, so every comparison table in the project is built
by the same function and the numbers are actually comparable.

`labels=` is passed explicitly to every metric. Without it, a CV fold whose
validation partition happens to be missing a class silently averages macro F1
over four classes instead of five, and the fold scores stop being comparable
to each other.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: training runs on servers with no display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_predictions(y_true, y_pred, id2label, model_name, inference_time_sec=None,
                          verbose=True):
    labels_sorted = sorted(id2label.keys())
    target_names = [id2label[i] for i in labels_sorted]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_sorted, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels_sorted,
                            zero_division=0)

    report = classification_report(
        y_true, y_pred, labels=labels_sorted, target_names=target_names,
        output_dict=True, zero_division=0,
    )

    if verbose:
        print(f"\n{model_name}")
        print(f"  accuracy {acc:.4f} | macro F1 {macro_f1:.4f} | weighted F1 {weighted_f1:.4f}")
        if inference_time_sec is not None:
            print(f"  inference {inference_time_sec:.3f}s over {len(y_true)} clauses")
        for name in target_names:
            row = report[name]
            print(f"    {name:<16} P={row['precision']:.3f} R={row['recall']:.3f} "
                  f"F1={row['f1-score']:.3f} n={int(row['support'])}")

    return {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "inference_time_sec": inference_time_sec,
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in target_names
        },
    }


def plot_confusion_matrix(y_true, y_pred, id2label, model_name, out_dir):
    labels_sorted = sorted(id2label.keys())
    target_names = [id2label[i] for i in labels_sorted]
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {model_name}")

    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    slug = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    out_path = os.path.join(out_dir, f"confusion_matrix_{slug}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_comparison_table(results, out_path, verbose=True):
    summary = [
        {
            "model": r["model"],
            "accuracy": round(r["accuracy"], 4),
            "macro_f1": round(r["macro_f1"], 4),
            "weighted_f1": round(r["weighted_f1"], 4),
            "inference_time_sec": (round(r["inference_time_sec"], 4)
                                    if r.get("inference_time_sec") is not None else None),
        }
        for r in results
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    df = pd.DataFrame(summary).sort_values("macro_f1", ascending=False)
    if verbose:
        print("\nmodel comparison (macro F1, descending)")
        print(df.to_string(index=False))
    return df


def save_per_class_table(results, out_path):
    """
    Per-class F1 for every model in one table. Worth reading alongside the
    macro average: Governing Law is a single CUAD category of near-identical
    boilerplate and scores far above the others, which pulls the macro
    average up for reasons that have nothing to do with the model.
    """
    rows = []
    for r in results:
        for cls, m in r["per_class"].items():
            rows.append({"model": r["model"], "class": cls, **m})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
