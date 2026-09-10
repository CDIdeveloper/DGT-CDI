"""Read-only investigation probe for the per-molecule prediction export.

Answers, in one run, everything needed to write
`scripts/export_oof_test_predictions.py` without guessing:

  A  what artifacts exist on this box
  B  raw parquet schema — SMILES column name, verbatim strings, duplicates
  C  what results/DGT_cv/dgt_cv_results.json records for the selected config
  D  which predictions.pt files are present (CV folds + 4 seeds)
  E  the payload schema of predictions.pt (probabilities vs logits, shapes)
  F  whether fold membership can be reconstructed from the parquet alone
  G  whether the harvested predictions reproduce paper.md §5.2b / §5.3
  H  the reference MPNN parquet's schema, for join compatibility

Nothing is written, no model is trained, no config is touched. Safe with
respect to CLAUDE.md gotcha #1 (never invokes main.py) and gotcha #2 (never
touches datasets/*/processed/).

Deliberately named `probe_*` and parked outside the `test_*.py` glob so a bare
`pytest` run does not collect it.

Run on the lab box, from the repo root, in the `dgt` env — either way works::

    python tests/probes/probe_oof_export.py
    pytest tests/probes/probe_oof_export.py -s -v -p no:cacheprovider

Under pytest, `-s` matters — the diagnostics go to stdout. Sections skip rather
than abort when their inputs are missing, so a partial box still yields a full
report.

Optional: point section H at the MPNN export to test the join key end to end::

    MPNN_PARQUET=s3://cdi-lab-workspaces/ts_project_1/data/biodegradation/GWU/predictions/mpnn_oof_test_qm_rdkit.parquet \
      python tests/probes/probe_oof_export.py
"""
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = 'BiodegNoInd-DGT-Pipeline-WithDesc-nongwu'
DATA_ROOT = REPO_ROOT / 'datasets' / 'biodeg_gwu_no_ind'
CV_JSON = REPO_ROOT / 'results' / 'DGT_cv' / 'dgt_cv_results.json'
CV_RUNS = REPO_ROOT / 'results' / 'DGT_cv' / CONFIG
SEED_RUNS = REPO_ROOT / 'results' / 'DGT' / CONFIG

K_FOLDS = 5
SEEDS = (0, 1, 2, 3)
# graphgps/loader/split_generator.py::CV_TRAIN_RANDOM_STATE — must not drift.
CV_RANDOM_STATE = 1

# Published values to reproduce (documents/projects/paper.md §5.2b, §5.3).
PAPER_CV_AUC = (0.8928, 0.0065)
PAPER_TEST_F1 = (0.8610, 0.0066)
PAPER_TEST_AUC = (0.9196, 0.0027)
PAPER_TEST_AP = (0.9269, 0.0051)


def _banner(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def _manifest():
    path = DATA_ROOT / 'raw' / 'manifest.json'
    if not path.is_file():
        pytest.skip(f'missing {path}')
    with open(path) as fh:
        return json.load(fh)


def _raw(split):
    path = DATA_ROOT / 'raw' / f'{split}.parquet'
    if not path.is_file():
        pytest.skip(f'missing {path}')
    return pd.read_parquet(path)


def _cv_cells():
    if not CV_JSON.is_file():
        pytest.skip(f'missing {CV_JSON}')
    with open(CV_JSON) as fh:
        blob = json.load(fh)
    return blob, {v['fold']: v for v in blob.get('cells', {}).values()
                  if v.get('config') == CONFIG}


def _fold_val_pred_path(fold, cells):
    """Where fold `fold` dumped its held-out (== val) predictions.

    Prefers the run_dir the CV runner recorded; falls back to the layout
    scripts/cv/dgt_cv_config.fold_out_dir builds.
    """
    if fold in cells and cells[fold].get('run_dir'):
        return Path(cells[fold]['run_dir']) / 'val' / 'predictions.pt'
    return CV_RUNS / f'fold{fold}' / CONFIG / '0' / 'val' / 'predictions.pt'


def _load_pred(path):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    y_true = np.asarray(blob['y_true']).reshape(-1)
    y_pred = np.asarray(blob['y_pred']).reshape(-1)
    return blob, y_true, y_pred


def _describe(path):
    if not path.is_file():
        return 'MISSING'
    st = path.stat()
    return f'{st.st_size:>9,} B  {pd.Timestamp(st.st_mtime, unit="s")}'


def _mean_pstd(values):
    a = np.asarray(values, dtype=float)
    return a.mean(), a.std()


def _binary_metrics(y_true, prob):
    return {
        'roc_auc': roc_auc_score(y_true, prob),
        'auprc': average_precision_score(y_true, prob),
        'f1@0.5': f1_score(y_true, (prob >= 0.5).astype(int), zero_division=0),
    }


def _cmp(label, got, expected):
    exp_mean, exp_std = expected
    delta = got - exp_mean
    flag = 'OK' if abs(delta) < 5e-4 else ('close' if abs(delta) < exp_std else 'DIFFERS')
    print(f'    {label:<34} {got:.4f}   paper {exp_mean:.4f} '
          f'(delta {delta:+.4f})  -> {flag}')


def test_a_inventory():
    _banner('A. Artifact inventory')
    print(f'repo root : {REPO_ROOT}')
    print(f'config    : {CONFIG}')
    for path in (DATA_ROOT / 'raw' / 'train.parquet',
                 DATA_ROOT / 'raw' / 'test.parquet',
                 DATA_ROOT / 'raw' / 'manifest.json',
                 CV_JSON):
        print(f'  {"OK " if path.is_file() else "MISS"}  {path.relative_to(REPO_ROOT)}')

    processed = DATA_ROOT / 'processed'
    print(f'\nprocessed caches in {processed}:')
    if processed.is_dir():
        for f in sorted(processed.iterdir()):
            print(f'    {f.name:<40} {f.stat().st_size:>13,} B')
    else:
        print('    (directory absent)')

    for label, root in (('single-split seed runs', SEED_RUNS), ('CV fold runs', CV_RUNS)):
        print(f'\n{label}: {root}')
        if root.is_dir():
            for f in sorted(root.iterdir()):
                print(f'    {f.name}')
        else:
            print('    (directory absent)')


def test_b_raw_parquet_schema():
    _banner('B. Raw parquet schema')
    man = _manifest()
    smiles_col = man['smiles_column']
    target_col = man['target_column']
    desc_cols = man['descriptor_columns']
    n_gwu = sum(1 for c in desc_cols if '_gwu' in c)
    print(f'manifest: smiles_column={smiles_col!r}  target_column={target_col!r}')
    print(f'          id_column_count={man.get("id_column_count")}  '
          f'desc_dim={man.get("desc_dim")}  task_type_hint={man.get("task_type_hint")!r}')
    print(f'          descriptors: {len(desc_cols)} total, {n_gwu} contain "_gwu", '
          f'{len(desc_cols) - n_gwu} do not  <- rdkit_fg arm expects 207')

    frames = {}
    for split in ('train', 'test'):
        df = _raw(split)
        frames[split] = df
        y = df[target_col].astype(float)
        s = df[smiles_col]
        print(f'\n{split}.parquet  shape={df.shape}')
        print(f'    {target_col}: positives={int((y == 1).sum())} '
              f'negatives={int((y == 0).sum())} other={int((~y.isin([0, 1])).sum())}')
        print(f'    {smiles_col}: dtype={s.dtype} nulls={int(s.isna().sum())} '
              f'unique={s.nunique()} / {len(s)}')
        dups = s[s.duplicated(keep=False)]
        if len(dups):
            print(f'    !! {len(dups)} duplicated SMILES rows — a string join '
                  f'would fan out. Examples: {sorted(set(dups))[:3]}')
        print(f'    first 3 verbatim: {[repr(v) for v in s.head(3).tolist()]}')
        print(f'    first 6 columns : {df.columns[:6].tolist()}')

    overlap = set(frames['train'][smiles_col]) & set(frames['test'][smiles_col])
    print(f'\ntrain/test SMILES overlap: {len(overlap)}  '
          f'(non-zero means (smiles, split) is required as the join key)')


def test_c_cv_results_json():
    _banner('C. results/DGT_cv/dgt_cv_results.json')
    blob, cells = _cv_cells()
    print('run-level: ' + '  '.join(
        f'{k}={blob.get(k)}' for k in
        ('dataset', 'k_folds', 'random_state', 'max_epoch',
         'metric_primary', 'metric_tiebreak', 'winner')))
    print(f'\ncells for {CONFIG}: {sorted(cells)}')
    for fold in sorted(cells):
        c = cells[fold]
        print(f'  fold {fold}: f1={c.get("f1"):.4f} @ep{c.get("f1_epoch")}   '
              f'auc={c.get("auc"):.4f} @ep{c.get("auc_epoch")}')
        print(f'          run_dir={c.get("run_dir")}')
    summary = blob.get('summaries', {}).get(CONFIG)
    if summary:
        print(f'\nrecorded summary: auc_mean={summary["auc_mean"]:.4f} '
              f'+- {summary["auc_std"]:.4f}   f1_mean={summary["f1_mean"]:.4f} '
              f'+- {summary["f1_std"]:.4f}')
        print(f'                  best_epoch_per_fold={summary["best_epoch_per_fold"]}')


def test_d_prediction_artifacts():
    _banner('D. predictions.pt presence')
    _, cells = _cv_cells()
    print('CV folds — held-out block is the val split under cv-train-5:')
    for fold in range(K_FOLDS):
        p = _fold_val_pred_path(fold, cells)
        print(f'  fold {fold} val : {_describe(p)}')
        print(f'            {p}')

    print('\nSingle-split seed runs:')
    for seed in SEEDS:
        for split in ('test', 'val'):
            p = SEED_RUNS / str(seed) / split / 'predictions.pt'
            print(f'  seed {seed} {split:<4}: {_describe(p)}')


def test_e_prediction_payload_schema():
    _banner('E. predictions.pt payload schema')
    _, cells = _cv_cells()
    targets = [(f'cv fold {f}', _fold_val_pred_path(f, cells)) for f in range(K_FOLDS)]
    targets += [(f'test seed {s}', SEED_RUNS / str(s) / 'test' / 'predictions.pt')
                for s in SEEDS]

    found = 0
    for label, path in targets:
        if not path.is_file():
            print(f'{label:<14} MISSING')
            continue
        found += 1
        blob, y_true, y_pred = _load_pred(path)
        extra = {k: v for k, v in blob.items() if k not in ('y_true', 'y_pred')}
        print(f'{label:<14} n={len(y_true):<5} keys={sorted(blob)}')
        print(f'{"":<14} y_true dtype={np.asarray(blob["y_true"]).dtype} '
              f'unique={np.unique(y_true).tolist()[:5]} positives={int(y_true.sum())}')
        print(f'{"":<14} y_pred dtype={np.asarray(blob["y_pred"]).dtype} '
              f'min={y_pred.min():.6f} max={y_pred.max():.6f} mean={y_pred.mean():.4f}')
        in_unit = float(y_pred.min()) >= 0.0 and float(y_pred.max()) <= 1.0
        print(f'{"":<14} in [0,1] -> {in_unit}  '
              f'({"probabilities (sigmoid applied)" if in_unit else "LOGITS — needs sigmoid"})')
        print(f'{"":<14} {extra}')
    if not found:
        pytest.skip('no predictions.pt found — sections F/G cannot run either')


def test_f_fold_reconstruction():
    _banner('F. Can fold membership be reconstructed from the parquet alone?')
    man = _manifest()
    df_train, df_test = _raw('train'), _raw('test')
    y_train = df_train[man['target_column']].astype(float).to_numpy()
    y_test = df_test[man['target_column']].astype(float).to_numpy()
    n_train = len(y_train)

    # Mirrors setup_cv_train_split: the CV pool is every non-test row, sorted by
    # dataset index. With no SMILES dropped at featurisation that is exactly
    # arange(n_train), i.e. train.parquet in file order.
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                          random_state=CV_RANDOM_STATE)
    folds = [val_pos for _, val_pos in skf.split(np.zeros(n_train), y_train)]
    print(f'train.parquet rows: {n_train}')
    print(f'fold sizes    : {[len(f) for f in folds]}  (paper §5.2b: 1053x4 + 1052)')
    print(f'fold positives: {[int(y_train[f].sum()) for f in folds]}  '
          f'sum={int(sum(y_train[f].sum() for f in folds))}  (paper §5.2b: 2466)')
    print(f'partition check: {sum(len(f) for f in folds)} assignments, '
          f'{len(set(np.concatenate(folds).tolist()))} distinct indices')

    _, cells = _cv_cells()
    print('\nLabel-vector match — proves position -> molecule without loading PyG:')
    verdicts = []
    for fold in range(K_FOLDS):
        path = _fold_val_pred_path(fold, cells)
        if not path.is_file():
            print(f'  fold {fold}: predictions.pt MISSING')
            continue
        _, y_true, _ = _load_pred(path)
        expected = y_train[folds[fold]]
        same_len = len(y_true) == len(expected)
        exact = same_len and bool(np.array_equal(y_true, expected))
        verdicts.append(exact)
        print(f'  fold {fold}: dumped n={len(y_true)} expected n={len(expected)} '
              f'-> {"EXACT MATCH" if exact else "MISMATCH"}')
        if same_len and not exact:
            n_diff = int((y_true != expected).sum())
            print(f'           {n_diff} labels differ; sorted multiset equal='
                  f'{np.array_equal(np.sort(y_true), np.sort(expected))} '
                  f'(equal => rows are permuted, not a different set)')

    for seed in SEEDS:
        path = SEED_RUNS / str(seed) / 'test' / 'predictions.pt'
        if not path.is_file():
            print(f'  test seed {seed}: predictions.pt MISSING')
            continue
        _, y_true, _ = _load_pred(path)
        exact = len(y_true) == len(y_test) and bool(np.array_equal(y_true, y_test))
        verdicts.append(exact)
        print(f'  test seed {seed}: dumped n={len(y_true)} expected n={len(y_test)} '
              f'-> {"EXACT MATCH" if exact else "MISMATCH"}')

    print('\nVERDICT: ' + (
        'parquet-order reconstruction is valid; the exporter needs no PyG load.'
        if verdicts and all(verdicts) else
        'reconstruction NOT confirmed — the exporter must load the PyG dataset '
        'and read per-Data .smiles instead.'))


def test_g_reproduce_published_metrics():
    _banner('G. Do the harvested predictions reproduce paper.md?')
    man = _manifest()
    df_train, df_test = _raw('train'), _raw('test')
    y_train = df_train[man['target_column']].astype(float).to_numpy()
    y_test = df_test[man['target_column']].astype(float).to_numpy()
    _, cells = _cv_cells()

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                          random_state=CV_RANDOM_STATE)
    folds = [val_pos for _, val_pos in skf.split(np.zeros(len(y_train)), y_train)]

    oof = np.full(len(y_train), np.nan)
    fold_aucs = []
    print('OOF-train, per fold (predictions are at the AUC-argmax checkpoint,')
    print('so per-fold ROC-AUC should match the CV json; per-fold F1 will not,')
    print('because fold_val_scores maximises F1 over epochs independently):')
    for fold in range(K_FOLDS):
        path = _fold_val_pred_path(fold, cells)
        if not path.is_file():
            print(f'  fold {fold}: MISSING')
            continue
        _, y_true, y_pred = _load_pred(path)
        if len(y_true) != len(folds[fold]):
            print(f'  fold {fold}: length mismatch, skipped')
            continue
        oof[folds[fold]] = y_pred
        m = _binary_metrics(y_true, y_pred)
        fold_aucs.append(m['roc_auc'])
        recorded = cells.get(fold, {}).get('auc')
        against = (f'json {recorded:.4f}, delta {m["roc_auc"] - recorded:+.4f}'
                   if recorded is not None else 'no json cell')
        print(f'  fold {fold}: roc_auc={m["roc_auc"]:.4f} ({against})  '
              f'auprc={m["auprc"]:.4f}  f1@0.5={m["f1@0.5"]:.4f}')

    if fold_aucs:
        mean, std = _mean_pstd(fold_aucs)
        print(f'\n  mean of per-fold ROC-AUC: {mean:.4f} +- {std:.4f}')
        _cmp('vs paper §5.2b CV ROC-AUC', mean, PAPER_CV_AUC)
    covered = int((~np.isnan(oof)).sum())
    print(f'\n  OOF coverage: {covered}/{len(y_train)} molecules scored')
    if covered == len(y_train):
        m = _binary_metrics(y_train, oof)
        print(f'  pooled OOF   : roc_auc={m["roc_auc"]:.4f}  '
              f'auprc={m["auprc"]:.4f}  f1@0.5={m["f1@0.5"]:.4f}')

    print('\nTest, per seed (threshold >= 0.5):')
    per_seed = {}
    for seed in SEEDS:
        path = SEED_RUNS / str(seed) / 'test' / 'predictions.pt'
        if not path.is_file():
            print(f'  seed {seed}: MISSING')
            continue
        _, y_true, y_pred = _load_pred(path)
        if len(y_true) != len(y_test):
            print(f'  seed {seed}: length mismatch, skipped')
            continue
        per_seed[seed] = y_pred
        m = _binary_metrics(y_true, y_pred)
        print(f'  seed {seed}: f1={m["f1@0.5"]:.4f}  roc_auc={m["roc_auc"]:.4f}  '
              f'auprc={m["auprc"]:.4f}')

    if len(per_seed) == len(SEEDS):
        stacked = np.vstack([per_seed[s] for s in SEEDS])
        print('\n  (a) per-seed metrics, then averaged — the comparable form:')
        for key, expected in (('f1@0.5', PAPER_TEST_F1),
                              ('roc_auc', PAPER_TEST_AUC),
                              ('auprc', PAPER_TEST_AP)):
            vals = [_binary_metrics(y_test, stacked[i])[key] for i in range(len(SEEDS))]
            mean, std = _mean_pstd(vals)
            _cmp(f'{key} mean +- {std:.4f}', mean, expected)

        print('\n  (b) metrics of the seed-mean probability — a 4-model ensemble:')
        m = _binary_metrics(y_test, stacked.mean(axis=0))
        print(f'    f1@0.5={m["f1@0.5"]:.4f}  roc_auc={m["roc_auc"]:.4f}  '
              f'auprc={m["auprc"]:.4f}')
        print('    (expected to sit slightly above (a) — that is the ensembling '
              'effect the per-seed columns exist to avoid crediting to DGT)')


def test_h_reference_mpnn_parquet():
    _banner('H. Reference MPNN export — join compatibility')
    ref = os.environ.get('MPNN_PARQUET')
    if not ref:
        pytest.skip('set MPNN_PARQUET=<path or s3:// uri> to run this section')
    df = pd.read_parquet(ref)
    print(f'{ref}\nshape={df.shape}')
    print(f'columns={df.columns.tolist()}')
    print(df.dtypes.to_string())
    if 'split' in df.columns:
        print(f'\nsplit counts: {df["split"].value_counts().to_dict()}')
    smiles_col = next((c for c in df.columns if 'smiles' in c.lower()), None)
    print(f'detected smiles column: {smiles_col!r}')
    if smiles_col:
        print(f'first 3 verbatim: {[repr(v) for v in df[smiles_col].head(3).tolist()]}')
        man = _manifest()
        ours = {s: set(_raw(s)[man['smiles_column']]) for s in ('train', 'test')}
        for split, mine in ours.items():
            if 'split' in df.columns:
                theirs = set(df.loc[df['split'] == split, smiles_col])
            else:
                theirs = set(df[smiles_col])
            print(f'  {split}: ours={len(mine)} theirs={len(theirs)} '
                  f'intersection={len(mine & theirs)} '
                  f'ours_unmatched={len(mine - theirs)}')
    print('\n=> the DGT export must use this column name and these exact strings.')


# Script mode, so the report does not depend on pytest collecting a file whose
# name is outside the `test_*.py` glob. pytest.skip raises a BaseException
# subclass, hence the explicit catch.
_SECTIONS = (
    test_a_inventory,
    test_b_raw_parquet_schema,
    test_c_cv_results_json,
    test_d_prediction_artifacts,
    test_e_prediction_payload_schema,
    test_f_fold_reconstruction,
    test_g_reproduce_published_metrics,
    test_h_reference_mpnn_parquet,
)


if __name__ == '__main__':
    for section in _SECTIONS:
        try:
            section()
        except pytest.skip.Exception as exc:
            print(f'  SKIPPED: {exc}')
        except Exception:
            traceback.print_exc()
    print('\n' + '=' * 78)
    print('probe complete — paste the whole output back')
    print('=' * 78)
