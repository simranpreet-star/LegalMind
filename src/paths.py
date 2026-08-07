"""
Central path config.

Three environments are supported and auto-detected:

  Kaggle            /kaggle/input for data (read-only), /kaggle/working for output
  Streamlit Cloud   repo is the working directory; models/ is committed alongside
  Local             repo root, resolved relative to this file

Everything resolves relative to the repository root rather than the caller's
working directory, so `python scripts/train_classical.py` behaves the same
from any directory.

Any path can be overridden with an environment variable of the same name
(LEGALMIND_DATA_DIR, LEGALMIND_MODELS_DIR, CUAD_JSON_PATH), which is how the
Kaggle notebook injects its own dataset locations without editing this file.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ON_KAGGLE = os.path.isdir("/kaggle/input")

if ON_KAGGLE:
    # Kaggle mounts attached datasets read-only; everything writable lives
    # under /kaggle/working and is persisted as notebook output.
    _DEFAULT_INPUT = "/kaggle/input"
    _DEFAULT_DATA = "/kaggle/working/data"
    _DEFAULT_MODELS = "/kaggle/working/models"
else:
    _DEFAULT_INPUT = os.path.join(REPO_ROOT, "data")
    _DEFAULT_DATA = os.path.join(REPO_ROOT, "data")
    _DEFAULT_MODELS = os.path.join(REPO_ROOT, "models")

INPUT_DIR = os.environ.get("LEGALMIND_INPUT_DIR", _DEFAULT_INPUT)
DATA_DIR = os.environ.get("LEGALMIND_DATA_DIR", _DEFAULT_DATA)
MODELS_DIR = os.environ.get("LEGALMIND_MODELS_DIR", _DEFAULT_MODELS)
REPORTS_DIR = os.environ.get("LEGALMIND_REPORTS_DIR", os.path.join(REPO_ROOT, "reports"))

CUAD_JSON_PATH = os.environ.get("CUAD_JSON_PATH", os.path.join(INPUT_DIR, "CUAD_v1.json"))
LABEL_MAPPING_PATH = os.path.join(MODELS_DIR, "label_mapping.json")


def ensure_dirs():
    """
    Create the writable directories. Called explicitly by training scripts
    rather than at import time -- importing a config module should not have
    filesystem side effects, and on Streamlit Cloud the app runs read-only
    against committed artifacts and must not try to mkdir anything.
    """
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        os.makedirs(d, exist_ok=True)
