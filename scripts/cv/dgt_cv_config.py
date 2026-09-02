"""Configuration for the DGT 5-fold cross-validation harness.

Mirrors `cv_config.py` in trans_learn's `kfold_cv/` module, which
documents/dgt_porting_guide.md §5 names as the template to replicate.

Everything the CV runner needs is here: which configs to evaluate, how many
folds, where results land. Edit this file rather than the runners.

Protocol constants come from dgt_porting_guide.md §2 and must not drift — they
are what make DGT's folds identical to the other models' folds on the same
train parquet in the same row order.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ─── Protocol constants (dgt_porting_guide.md §2) ────────────────────────────
# Fold count and RNG seed are also hard-coded in
# graphgps/loader/split_generator.py::setup_cv_train_split as
# CV_TRAIN_RANDOM_STATE; K_FOLDS here must match the `cv-train-<k>` split_mode
# string the runner builds. Changing either without the other is a silent bug.
K_FOLDS = 5
RANDOM_STATE = 1

# One training seed per fold: folds supply the dispersion, so per-fold seeds
# would only multiply cost. §2 specifies a fresh model per fold.
SEED = 0

# Selection metric (primary) and its tiebreak, per §2.
METRIC_PRIMARY = 'f1'
METRIC_TIEBREAK = 'auc'

# ─── What to evaluate ────────────────────────────────────────────────────────
# Config basenames under configs/biodegradability/. Each is run K_FOLDS times.
DATASET = 'biodeg_gwu_no_ind'
CONFIGS = [
    'BiodegNoInd-DGT-Pipeline',                    # none (graph only)
    'BiodegNoInd-DGT-Pipeline-WithDesc-gwu',       # qm (40)
    'BiodegNoInd-DGT-Pipeline-WithDesc-nongwu',    # rdkit_fg (207)
    'BiodegNoInd-DGT-Pipeline-WithDesc',           # qm_rdkit (247)
]
CONFIG_DIR = REPO_ROOT / 'configs' / 'biodegradability'

MAX_EPOCH = 50

# ─── Output ──────────────────────────────────────────────────────────────────
# Kept out of results/DGT/ so CV runs never collide with the single-split runs
# (and so main.py's per-seed directory wipe cannot touch them).
CV_OUT_DIR = REPO_ROOT / 'results' / 'DGT_cv'
RESULTS_JSON = CV_OUT_DIR / 'dgt_cv_results.json'
RESULTS_MD = CV_OUT_DIR / 'dgt_cv_results.md'


def split_mode() -> str:
    """The `dataset.split_mode` string that selects train+val-only k-folding."""
    return f'cv-train-{K_FOLDS}'


def config_path(name: str) -> Path:
    path = CONFIG_DIR / f'{name}.yaml'
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def fold_out_dir(name: str, fold: int) -> Path:
    """Where main.py writes one (config, fold) run."""
    return CV_OUT_DIR / name / f'fold{fold}'
