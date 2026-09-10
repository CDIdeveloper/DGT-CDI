"""Emit per-molecule DGT predictions for the biodegradability cross-model analysis.

Harvests probabilities that already exist on disk. Nothing is trained, no
configuration is read, no selection decision is made — `main.py` is never
invoked (CLAUDE.md gotcha #1) and `datasets/*/processed/` is never touched
(gotcha #2).

Two sources, both for the selected configuration
`BiodegNoInd-DGT-Pipeline-WithDesc-nongwu` (`rdkit_fg`, 207 descriptors):

  train rows (5264)  out-of-fold probabilities from the stratified 5-fold CV
                     behind paper.md §5.2b/§6.2. Under `split_mode: cv-train-5`
                     a fold's held-out block *is* that run's validation split
                     (split_generator.py::setup_cv_train_split), and
                     dgt_train.py dumps per-sample val predictions at the
                     best-val checkpoint. So each fold's val/predictions.pt is
                     its OOF block, and the five partition the train parquet:
                     every training molecule scored exactly once, always by a
                     model that did not fit it.

  test rows (278)    the 4 seeds behind §5.3, each at its own best-val
                     checkpoint. NOT the train+val deployment retrain.
                     `prob` is the seed mean; `prob_seed0..3` are the
                     individuals, because the seed mean is a 4-model ensemble
                     while the HGB and MPNN it is compared against are single
                     models — the downstream comparison must average metrics,
                     not probabilities.

Row identity is reconstructed from the raw parquet rather than stored: the PyG
dataset featurises train.parquet then test.parquet in file order, and the eval
loaders do not shuffle, so dataset index == parquet row. That is *verified*,
not assumed — every dump's label vector is checked elementwise against the
parquet's, and the script aborts on any mismatch.

Usage (lab box, repo root, `dgt` env):
    python scripts/export_oof_test_predictions.py

Writes results/predictions/dgt_oof_test_rdkit_fg.parquet (gitignored) and
prints the sanity-check report.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'BiodegNoInd-DGT-Pipeline-WithDesc-nongwu'
RAW_DIR = REPO_ROOT / 'datasets' / 'biodeg_gwu_no_ind' / 'raw'
CV_JSON = REPO_ROOT / 'results' / 'DGT_cv' / 'dgt_cv_results.json'
CV_RUNS = REPO_ROOT / 'results' / 'DGT_cv' / CONFIG
SEED_RUNS = REPO_ROOT / 'results' / 'DGT' / CONFIG
OUT_PATH = REPO_ROOT / 'results' / 'predictions' / 'dgt_oof_test_rdkit_fg.parquet'
S3_DEST = ('s3://cdi-lab-workspaces/ts_project_1/data/biodegradation/GWU/'
           'predictions/')

K_FOLDS = 5
SEEDS = (0, 1, 2, 3)
# graphgps/loader/split_generator.py::CV_TRAIN_RANDOM_STATE. Changing this here
# without changing it there silently re-folds the data.
CV_RANDOM_STATE = 1

# Expected row counts and published values, asserted/compared below so a silent
# regression in any upstream artifact shows up as a failed check.
N_TRAIN, N_TEST = 5264, 278
POS_TRAIN, POS_TEST = 2466, 144
PAPER_CV_AUC = (0.8928, 0.0065)      # §5.2b
PAPER_TEST_F1 = (0.8610, 0.0066)     # §5.3
PAPER_TEST_AUC = (0.9196, 0.0027)    # §5.3
PAPER_TEST_AP = (0.9269, 0.0051)     # §5.3


def _banner(title):
    print('\n' + '-' * 72)
    print(title)
    print('-' * 72)


def _load_pred(path):
    """Return (y_true, y_pred, best_epoch). Dumps are float16 under autocast."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. This export harvests existing runs; it does not "
            f"train. Re-run the source sweep if the artifact is genuinely gone."
        )
    blob = torch.load(path, map_location='cpu', weights_only=False)
    return (np.asarray(blob['y_true']).reshape(-1).astype(np.float64),
            np.asarray(blob['y_pred']).reshape(-1).astype(np.float64),
            blob['best_epoch'])


def _fold_val_pred_path(fold, cells):
    """Where fold `fold` dumped its held-out block, per the CV runner's record."""
    if fold in cells and cells[fold].get('run_dir'):
        return Path(cells[fold]['run_dir']) / 'val' / 'predictions.pt'
    return CV_RUNS / f'fold{fold}' / CONFIG / '0' / 'val' / 'predictions.pt'


def _check_labels(label, got, expected):
    """Abort unless a dump's label vector matches the parquet's, elementwise.

    This is what pins position -> molecule. A shuffled or mis-indexed dump
    cannot survive it: across 1053 rows at ~47% positives, agreement by chance
    is not a possibility worth reasoning about.
    """
    if len(got) != len(expected) or not np.array_equal(got, expected):
        raise RuntimeError(
            f"{label}: label vector does not match the parquet "
            f"(dumped n={len(got)}, expected n={len(expected)}). The "
            f"position -> molecule mapping is not valid; do not use the output."
        )


def _mean_pstd(values):
    a = np.asarray(values, dtype=float)
    return a.mean(), a.std()


def _binary_metrics(y_true, prob):
    return {
        'roc_auc': roc_auc_score(y_true, prob),
        'auprc': average_precision_score(y_true, prob),
        'f1@0.5': f1_score(y_true, (prob >= 0.5).astype(int), zero_division=0),
    }


def _report(label, got, paper):
    mean, std = paper
    print(f'    {label:<26} {got:.4f}    paper {mean:.4f} +- {std:.4f}  '
          f'(delta {got - mean:+.4f})')


def _cv_cells():
    with open(CV_JSON) as fh:
        blob = json.load(fh)
    return {v['fold']: v for v in blob.get('cells', {}).values()
            if v.get('config') == CONFIG}


def oof_train_probs(y_train, folds):
    """Scatter each fold's held-out probabilities into one OOF vector."""
    cells = _cv_cells()
    oof = np.full(len(y_train), np.nan)
    for fold in range(K_FOLDS):
        path = _fold_val_pred_path(fold, cells)
        y_true, y_pred, best_epoch = _load_pred(path)
        _check_labels(f'CV fold {fold}', y_true, y_train[folds[fold]])
        oof[folds[fold]] = y_pred
        print(f'  fold {fold}: {len(y_pred):>4} held-out molecules, '
              f'best-val checkpoint epoch {best_epoch}')
    if np.isnan(oof).any():
        raise RuntimeError(f"{int(np.isnan(oof).sum())} training molecules "
                           f"were never scored; the folds do not partition.")
    return oof


def test_seed_probs(y_test):
    """One probability vector per seed, at that seed's best-val checkpoint."""
    per_seed = {}
    for seed in SEEDS:
        path = SEED_RUNS / str(seed) / 'test' / 'predictions.pt'
        y_true, y_pred, best_epoch = _load_pred(path)
        _check_labels(f'test seed {seed}', y_true, y_test)
        per_seed[seed] = y_pred
        print(f'  seed {seed}: {len(y_pred):>4} test molecules, '
              f'best-val checkpoint epoch {best_epoch}')
    return per_seed


def main():
    with open(RAW_DIR / 'manifest.json') as fh:
        manifest = json.load(fh)
    smiles_col, target_col = manifest['smiles_column'], manifest['target_column']
    df_train = pd.read_parquet(RAW_DIR / 'train.parquet')
    df_test = pd.read_parquet(RAW_DIR / 'test.parquet')
    y_train = df_train[target_col].astype(float).to_numpy()
    y_test = df_test[target_col].astype(float).to_numpy()

    # Mirrors setup_cv_train_split: the CV pool is every non-test row in dataset
    # order, which (nothing dropped at featurisation) is train.parquet in file
    # order. Verified by _check_labels below.
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                          random_state=CV_RANDOM_STATE)
    folds = [val_pos for _, val_pos in skf.split(np.zeros(len(y_train)), y_train)]
    fold_of = np.empty(len(y_train), dtype=np.int64)
    for i, idx in enumerate(folds):
        fold_of[idx] = i

    _banner('Harvesting out-of-fold probabilities (5-fold CV on train)')
    oof = oof_train_probs(y_train, folds)
    _banner('Harvesting test probabilities (4 seeds)')
    per_seed = test_seed_probs(y_test)
    stacked = np.vstack([per_seed[s] for s in SEEDS])

    train_rows = pd.DataFrame({
        'smiles': df_train[smiles_col].to_numpy(),   # verbatim, never re-canonicalised
        'split': 'train',
        'true': y_train.astype(np.int64),
        'prob': oof,
        # Per-seed columns are test-only: OOF rows are one model per fold.
        **{f'prob_seed{s}': np.full(len(df_train), np.nan) for s in SEEDS},
        'fold': fold_of,
    })
    test_rows = pd.DataFrame({
        'smiles': df_test[smiles_col].to_numpy(),
        'split': 'test',
        'true': y_test.astype(np.int64),
        'prob': stacked.mean(axis=0),
        **{f'prob_seed{s}': stacked[i] for i, s in enumerate(SEEDS)},
        'fold': np.full(len(df_test), -1, dtype=np.int64),
    })
    out = pd.concat([train_rows, test_rows], ignore_index=True)

    _banner('1. Structure')
    n_tr = int((out['split'] == 'train').sum())
    n_te = int((out['split'] == 'test').sum())
    print(f'  rows                  {len(out)} = {n_tr} train + {n_te} test'
          f'   {"OK" if (len(out), n_tr, n_te) == (N_TRAIN + N_TEST, N_TRAIN, N_TEST) else "FAIL"}')
    print(f'  NaN in prob           {int(out["prob"].isna().sum())}'
          f'   {"OK" if not out["prob"].isna().any() else "FAIL"}')
    uniq = out.loc[out['split'] == 'train', 'smiles'].nunique()
    print(f'  train SMILES unique   {uniq}/{n_tr}   {"OK" if uniq == n_tr else "FAIL"}')
    seed_cols = [f'prob_seed{s}' for s in SEEDS]
    print(f'  prob_seed* on train   all NaN: '
          f'{bool(out.loc[out["split"] == "train", seed_cols].isna().all().all())}')
    print(f'  prob_seed* on test    no NaN : '
          f'{bool(out.loc[out["split"] == "test", seed_cols].notna().all().all())}')

    _banner('2. Join against the source parquet on (smiles, split)')
    source = pd.concat([
        pd.DataFrame({'smiles': df_train[smiles_col].to_numpy(), 'split': 'train'}),
        pd.DataFrame({'smiles': df_test[smiles_col].to_numpy(), 'split': 'test'}),
    ], ignore_index=True)
    merged = out.merge(source, on=['smiles', 'split'], how='left', indicator=True)
    matched = int((merged['_merge'] == 'both').sum())
    print(f'  matched               {matched}/{len(out)} '
          f'({100 * matched / len(out):.1f}%)   '
          f'{"OK" if matched == len(out) else "FAIL"}')
    # A duplicated (smiles, split) in the source would fan the merge out, which
    # is the failure mode a string-keyed join actually has downstream.
    print(f'  no fan-out            {len(merged)} rows out of {len(out)}   '
          f'{"OK" if len(merged) == len(out) else "FAIL"}')
    if matched != len(out) or len(merged) != len(out):
        raise RuntimeError('SMILES do not round-trip 1:1 against the source '
                           'parquet; the downstream join would be wrong.')

    _banner('3. Class balance')
    for split, expected in (('train', POS_TRAIN), ('test', POS_TEST)):
        got = int(out.loc[out['split'] == split, 'true'].sum())
        print(f'  {split:<6} positives       {got} (expected {expected})   '
              f'{"OK" if got == expected else "FAIL"}')

    _banner('4. Metrics recomputed FROM THE PARQUET (threshold prob >= 0.5)')
    tr = out[out['split'] == 'train']
    m = _binary_metrics(tr['true'].to_numpy(), tr['prob'].to_numpy())
    print('  OOF-train, 5264 molecules, one single model per fold:')
    print(f'    pooled over all folds      roc_auc={m["roc_auc"]:.4f}  '
          f'auprc={m["auprc"]:.4f}  f1@0.5={m["f1@0.5"]:.4f}')
    fold_aucs = [_binary_metrics(g['true'].to_numpy(), g['prob'].to_numpy())['roc_auc']
                 for _, g in tr.groupby('fold')]
    mean, std = _mean_pstd(fold_aucs)
    print('    per-fold roc_auc           ' + ', '.join(f'{v:.4f}' for v in fold_aucs))
    _report(f'mean +- {std:.4f}', mean, PAPER_CV_AUC)

    te = out[out['split'] == 'test']
    y_te = te['true'].to_numpy()
    print('\n  Test, 278 molecules:')
    print('    (a) per seed, then averaged — the single-model form, use this '
          'for cross-model comparison:')
    for key, paper in (('f1@0.5', PAPER_TEST_F1), ('roc_auc', PAPER_TEST_AUC),
                       ('auprc', PAPER_TEST_AP)):
        vals = [_binary_metrics(y_te, te[c].to_numpy())[key] for c in seed_cols]
        mean, std = _mean_pstd(vals)
        _report(f'{key} mean +- {std:.4f}', mean, paper)
    m = _binary_metrics(y_te, te['prob'].to_numpy())
    print('    (b) from the seed-mean `prob` — a 4-model ensemble, expected '
          'slightly higher:')
    print(f'      f1@0.5={m["f1@0.5"]:.4f}  roc_auc={m["roc_auc"]:.4f}  '
          f'auprc={m["auprc"]:.4f}')

    _banner('5. Numeric precision')
    probs = out[['prob'] + seed_cols].to_numpy()
    probs = probs[~np.isnan(probs)]
    print(f'  Dumps are float16 (torch.autocast in dgt_train.py) widened to '
          f'float64 losslessly.')
    print(f'  exactly 0.0           {int((probs == 0.0).sum())}')
    print(f'  exactly 1.0           {int((probs == 1.0).sum())}')
    print('  Ranking metrics (ROC-AUC, AUPRC) are unaffected. Log-loss and the '
          'extreme bins of a\n  calibration curve are: saturated values give '
          'an infinite log-loss term. Clip before\n  computing either.')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    _banner('Written')
    print(f'  {OUT_PATH}')
    print(f'  shape={out.shape}  columns={out.columns.tolist()}')
    print(f'\n  Upload with:\n    aws s3 cp {OUT_PATH} {S3_DEST}')


if __name__ == '__main__':
    main()
