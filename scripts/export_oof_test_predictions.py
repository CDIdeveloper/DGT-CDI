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

Any ablation arm can be exported. Exporting a non-selected arm is an
**ablation-context harvest, not a re-selection**: the recorded selection stays
`rdkit_fg` (§6, §6.2) and the headline test figures in §5.3 are unchanged.
Because the fold assignment depends only on the train parquet — not on the arm —
two arms' exports are paired molecule by molecule, which `2b` verifies.

Usage (lab box, repo root, `dgt` env):
    python scripts/export_oof_test_predictions.py                    # rdkit_fg
    python scripts/export_oof_test_predictions.py --arm qm_rdkit     # matches MPNN
    python scripts/export_oof_test_predictions.py --arm none         # graph only

Writes results/predictions/dgt_oof_test_<arm>.parquet (gitignored) and prints
the sanity-check report.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / 'datasets' / 'biodeg_gwu_no_ind' / 'raw'
CV_JSON = REPO_ROOT / 'results' / 'DGT_cv' / 'dgt_cv_results.json'
CV_ROOT = REPO_ROOT / 'results' / 'DGT_cv'
SEED_ROOT = REPO_ROOT / 'results' / 'DGT'
PRED_DIR = REPO_ROOT / 'results' / 'predictions'
S3_DEST = ('s3://cdi-lab-workspaces/ts_project_1/data/biodegradation/GWU/'
           'predictions/')

K_FOLDS = 5
SEEDS = (0, 1, 2, 3)
# graphgps/loader/split_generator.py::CV_TRAIN_RANDOM_STATE. Changing this here
# without changing it there silently re-folds the data.
CV_RANDOM_STATE = 1

# Expected row counts, asserted below so a silent regression in any upstream
# artifact shows up as a failed check. Identical for every arm — the arms differ
# only in which descriptors reach the model, never in which molecules do.
N_TRAIN, N_TEST = 5264, 278
POS_TRAIN, POS_TEST = 2466, 144

# One entry per ablation arm. `cv_auc` is that arm's published §5.2b figure.
# `test` is populated for the SELECTED arm only, because §5.3 reads the test
# split for that configuration alone; exporting any other arm is an
# ablation-context harvest and changes no selection (§6, §6.2, §10.2).
ARMS = {
    'rdkit_fg': {
        'config': 'BiodegNoInd-DGT-Pipeline-WithDesc-nongwu',
        'about': '207 RDKit/functional-group descriptors — the selected arm (§6)',
        'cv_auc': (0.8928, 0.0065),
        'test': {'f1@0.5': (0.8610, 0.0066),
                 'roc_auc': (0.9196, 0.0027),
                 'auprc': (0.9269, 0.0051)},
    },
    'qm_rdkit': {
        'config': 'BiodegNoInd-DGT-Pipeline-WithDesc',
        'about': '247 descriptors (40 QM + 207 RDKit/fg) — matches the MPNN baseline',
        'cv_auc': (0.8925, 0.0064),
        'test': None,
    },
    'none': {
        'config': 'BiodegNoInd-DGT-Pipeline',
        'about': 'graph only, no descriptor channel',
        'cv_auc': (0.8893, 0.0051),
        'test': None,
    },
}


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


def _fold_val_pred_path(fold, cells, config):
    """Where fold `fold` dumped its held-out block, per the CV runner's record."""
    if fold in cells and cells[fold].get('run_dir'):
        return Path(cells[fold]['run_dir']) / 'val' / 'predictions.pt'
    return CV_ROOT / config / f'fold{fold}' / config / '0' / 'val' / 'predictions.pt'


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


def _source_paths(config):
    """Every dump this export reads, as (label, path): 5 CV folds + 4 seeds."""
    with open(CV_JSON) as fh:
        blob = json.load(fh)
    cells = {v['fold']: v for v in blob.get('cells', {}).values()
             if v.get('config') == config}
    fold_paths = [(f'CV fold {f}', _fold_val_pred_path(f, cells, config))
                  for f in range(K_FOLDS)]
    seed_paths = [(f'test seed {s}',
                   SEED_ROOT / config / str(s) / 'test' / 'predictions.pt')
                  for s in SEEDS]
    return fold_paths, seed_paths


def oof_train_probs(y_train, fold_index, fold_paths):
    """Scatter each fold's held-out probabilities into one OOF vector."""
    oof = np.full(len(y_train), np.nan)
    for fold, (label, path) in enumerate(fold_paths):
        y_true, y_pred, best_epoch = _load_pred(path)
        _check_labels(label, y_true, y_train[fold_index[fold]])
        oof[fold_index[fold]] = y_pred
        print(f'  fold {fold}: {len(y_pred):>4} held-out molecules, '
              f'best-val checkpoint epoch {best_epoch}')
    if np.isnan(oof).any():
        raise RuntimeError(f"{int(np.isnan(oof).sum())} training molecules "
                           f"were never scored; the folds do not partition.")
    return oof


def test_seed_probs(y_test, seed_paths):
    """One probability vector per seed, at that seed's best-val checkpoint."""
    per_seed = {}
    for seed, (label, path) in zip(SEEDS, seed_paths):
        y_true, y_pred, best_epoch = _load_pred(path)
        _check_labels(label, y_true, y_test)
        per_seed[seed] = y_pred
        print(f'  seed {seed}: {len(y_pred):>4} test molecules, '
              f'best-val checkpoint epoch {best_epoch}')
    return per_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm', default='rdkit_fg', choices=sorted(ARMS),
                    help="Ablation arm to export. Default: rdkit_fg, the "
                         "selected configuration. Exporting another arm is an "
                         "ablation-context harvest, not a re-selection.")
    args = ap.parse_args()
    arm = ARMS[args.arm]
    config = arm['config']
    out_path = PRED_DIR / f'dgt_oof_test_{args.arm}.parquet'
    print(f'arm     {args.arm} — {arm["about"]}')
    print(f'config  {config}')
    print(f'output  {out_path}')

    with open(RAW_DIR / 'manifest.json') as fh:
        manifest = json.load(fh)
    smiles_col, target_col = manifest['smiles_column'], manifest['target_column']
    df_train = pd.read_parquet(RAW_DIR / 'train.parquet')
    df_test = pd.read_parquet(RAW_DIR / 'test.parquet')
    y_train = df_train[target_col].astype(float).to_numpy()
    y_test = df_test[target_col].astype(float).to_numpy()

    # Mirrors setup_cv_train_split: the CV pool is every non-test row in dataset
    # order, which (nothing dropped at featurisation) is train.parquet in file
    # order. Verified by _check_labels below. Independent of the arm, which is
    # what makes two arms' exports paired molecule by molecule.
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                          random_state=CV_RANDOM_STATE)
    fold_index = [val_pos for _, val_pos in
                  skf.split(np.zeros(len(y_train)), y_train)]
    fold_of = np.empty(len(y_train), dtype=np.int64)
    for i, idx in enumerate(fold_index):
        fold_of[idx] = i

    _banner('0. Preflight — every source dump this export reads')
    fold_paths, seed_paths = _source_paths(config)
    missing = []
    for label, path in fold_paths + seed_paths:
        ok = path.is_file()
        missing += [] if ok else [path]
        print(f'  {"OK  " if ok else "MISS"} {label:<12} {path}')
    if missing:
        raise SystemExit(
            f"\n{len(missing)} source dump(s) missing for arm '{args.arm}'. "
            f"This export harvests existing runs and does not train; the "
            f"listed runs must be present (or re-run) first."
        )

    _banner('Harvesting out-of-fold probabilities (5-fold CV on train)')
    oof = oof_train_probs(y_train, fold_index, fold_paths)
    _banner('Harvesting test probabilities (4 seeds)')
    per_seed = test_seed_probs(y_test, seed_paths)
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

    _banner('2b. Cross-check against arms already exported')
    others = sorted(p for p in PRED_DIR.glob('dgt_oof_test_*.parquet')
                    if p != out_path)
    if not others:
        print('  (no other arm exported yet — nothing to cross-check)')
    for other in others:
        ref = pd.read_parquet(other, columns=['smiles', 'split', 'true', 'fold'])
        pair = out.merge(ref, on=['smiles', 'split'], how='inner',
                         suffixes=('', '_ref'))
        joined = len(pair) == len(out)
        same_true = bool((pair['true'] == pair['true_ref']).all())
        same_fold = bool((pair['fold'] == pair['fold_ref']).all())
        print(f'  vs {other.name}')
        print(f'    joined 1:1            {len(pair)}/{len(out)}   '
              f'{"OK" if joined else "FAIL"}')
        print(f'    `true` identical      {same_true}   '
              f'{"OK" if same_true else "FAIL"}')
        print(f'    `fold` identical      {same_fold}   '
              f'{"OK" if same_fold else "FAIL"}   <- folds coincide, so the '
              f'arms are paired per molecule')
        if not (joined and same_true and same_fold):
            raise RuntimeError(
                f'{other.name} disagrees with this export on keys, labels or '
                f'fold assignment; a paired comparison would be invalid.')

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
    _report(f'mean +- {std:.4f}', mean, arm['cv_auc'])

    te = out[out['split'] == 'test']
    y_te = te['true'].to_numpy()
    print('\n  Test, 278 molecules:')
    print('    (a) per seed, then averaged — the single-model form, use this '
          'for cross-model comparison:')
    for key in ('f1@0.5', 'roc_auc', 'auprc'):
        vals = [_binary_metrics(y_te, te[c].to_numpy())[key] for c in seed_cols]
        mean, std = _mean_pstd(vals)
        label = f'{key} mean +- {std:.4f}'
        if arm['test'] is not None:
            _report(label, mean, arm['test'][key])
        else:
            print(f'    {label:<26} {mean:.4f}    (no published value — §5.3 '
                  f'reads test for the selected arm only)')
    m = _binary_metrics(y_te, te['prob'].to_numpy())
    print('    (b) from the seed-mean `prob` — a 4-model ensemble, expected '
          'slightly higher:')
    print(f'      f1@0.5={m["f1@0.5"]:.4f}  roc_auc={m["roc_auc"]:.4f}  '
          f'auprc={m["auprc"]:.4f}')

    _banner('5. Numeric precision')
    probs = out[['prob'] + seed_cols].to_numpy()
    probs = probs[~np.isnan(probs)]
    print('  Dumps are float16 (torch.autocast in dgt_train.py) widened to '
          'float64 losslessly.')
    print(f'  exactly 0.0           {int((probs == 0.0).sum())}')
    print(f'  exactly 1.0           {int((probs == 1.0).sum())}')
    print('  Ranking metrics (ROC-AUC, AUPRC) are unaffected. Log-loss and the '
          'extreme bins of a\n  calibration curve are: saturated values give '
          'an infinite log-loss term. Clip before\n  computing either.')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    _banner('Written')
    print(f'  {out_path}')
    print(f'  shape={out.shape}  columns={out.columns.tolist()}')
    print(f'\n  Upload with:\n    aws s3 cp {out_path} {S3_DEST}')


if __name__ == '__main__':
    main()
