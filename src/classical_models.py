"""
Classical baselines for clause classification: Logistic Regression, linear
SVM, Random Forest and XGBoost, all over the same TF-IDF features so the
comparison is fair.

Two things here are deliberate and are the difference between an honest score
and an inflated one:

1. Every model is an imblearn Pipeline of (tfidf -> [resampler] -> clf), and
   the pipeline is what gets handed to GridSearchCV. So the vectoriser is
   fitted on each training fold alone, and any resampling happens after the
   fold boundary is drawn. Fitting TF-IDF on the whole training set, or
   oversampling before the search, both leak: the first leaks document
   frequencies, the second puts exact duplicate rows on both sides of every
   inner CV split and quietly inflates the selected hyperparameters.

2. The search uses GroupKFold on doc_id, so hyperparameter selection is
   document-disjoint for the same reason the final test split is.

Balancing strategy per model: the linear models and the forest take
class_weight, which needs no row duplication at all. XGBoost has no usable
class_weight through the sklearn API under grid search, so it gets a
RandomOverSampler as an explicit pipeline step -- correct precisely because
the pipeline confines it to the training side of each fold.
"""

import time

import joblib
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

RANDOM_STATE = 42


def build_tfidf_vectorizer(max_features=15000, ngram_range=(1, 2)):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        stop_words="english",
    )


def _pipeline(estimator, oversample=False):
    steps = [("tfidf", build_tfidf_vectorizer())]
    if oversample:
        # imblearn pipelines apply samplers on fit() only, never on
        # predict()/transform(), which is what keeps this out of the
        # evaluation path.
        steps.append(("resample", RandomOverSampler(random_state=RANDOM_STATE)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def _fit(pipeline, param_grid, texts, y, groups, tune=True, n_splits=5, verbose=0):
    """
    Fits a pipeline, optionally grid-searching over param_grid with
    document-grouped CV. Returns (fitted_estimator, best_params).
    """
    if not tune or not param_grid:
        pipeline.fit(texts, y)
        return pipeline, {}

    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=splits),
        n_jobs=-1,
        verbose=verbose,
    )
    search.fit(texts, y, groups=groups)
    return search.best_estimator_, search.best_params_


def train_logistic_regression(texts, y, groups, tune=True):
    est = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    return _fit(_pipeline(est), {"clf__C": [0.1, 1.0, 10.0]}, texts, y, groups, tune)


def train_svm(texts, y, groups, tune=True):
    # LinearSVC rather than SVC(kernel="linear", probability=True): identical
    # decision function, but liblinear scales to this feature count in
    # seconds where libsvm's O(n^2) fit plus internal Platt scaling took
    # hours. It has no predict_proba, so it is a benchmark-table model only
    # and is not offered for deployment (see MODELS_WITH_PROBA).
    est = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    return _fit(_pipeline(est), {"clf__C": [0.1, 1.0, 10.0]}, texts, y, groups, tune)


def train_random_forest(texts, y, groups, tune=True):
    est = RandomForestClassifier(
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
    )
    grid = {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 30]}
    return _fit(_pipeline(est), grid, texts, y, groups, tune)


def train_xgboost(texts, y, groups, tune=True):
    # num_class is inferred from y by the sklearn wrapper; passing it
    # explicitly alongside multi:softprob errors on xgboost >= 2.0.
    est = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [4, 6],
        "clf__learning_rate": [0.05, 0.1],
    }
    return _fit(_pipeline(est, oversample=True), grid, texts, y, groups, tune)


TRAINERS = {
    "Logistic Regression": train_logistic_regression,
    "SVM": train_svm,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
}

# Models exposing calibrated-enough predict_proba for the abstention
# threshold. The app restricts its model picker to these.
MODELS_WITH_PROBA = ("Logistic Regression", "Random Forest", "XGBoost")


def predict_with_abstention(model, texts, threshold=0.6, unclassified_id=-1):
    """
    Returns (predictions, confidences) with anything below `threshold`
    replaced by `unclassified_id`.

    This exists because the five target classes cover roughly a fifth of
    CUAD's categories. Most clauses in a real uploaded contract belong to
    none of them, and a plain argmax will assign one anyway with no signal
    that it is out of scope. See scripts/evaluate_out_of_scope.py for what
    this threshold actually buys.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{type(model).__name__} has no predict_proba; abstention needs a "
            f"probabilistic model (one of {MODELS_WITH_PROBA})"
        )
    probs = model.predict_proba(texts)
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    preds = np.where(confidences < threshold, unclassified_id, preds)
    return preds, confidences


def timed_predict(model, texts):
    start = time.perf_counter()
    preds = model.predict(texts)
    return preds, time.perf_counter() - start


def save_model(obj, path):
    joblib.dump(obj, path, compress=3)


def load_model(path):
    return joblib.load(path)
