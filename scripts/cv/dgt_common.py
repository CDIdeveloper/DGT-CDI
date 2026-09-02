"""Shared helpers for the DGT cross-validation harness.

Counterpart to `mpnn_common.py` in trans_learn's `kfold_cv/`
(documents/dgt_porting_guide.md §5): run one fold, read back its validation
score, and aggregate folds into a CV summary.

A fold is executed by subprocessing `main.py` rather than importing the
training stack, for the same reason `scripts/retrain_on_trainval.py` does: the
GraphGym config is global mutable state, so running several configurations in
one process would leak settings between them. One process per fold also means a
crashed fold cannot corrupt the others' results.
"""
import json
import statistics
import subprocess
import sys

import dgt_cv_config as C

# metric_agg -> selector for "best validation epoch"
_AGG = {'argmax': max, 'argmin': min}


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_fold(config_name, fold, extra_overrides=None):
    """Train one (config, fold). Returns the per-seed run dir main.py wrote.

    The fold is selected purely through config overrides — `split_mode:
    cv-train-<k>` plus `dataset.split_index: <fold>` — so no YAML is copied or
    mutated, and the four configs stay byte-identical to the ones used for the
    single-split runs.
    """
    out_dir = C.fold_out_dir(config_name, fold)
    cmd = [
        sys.executable, 'main.py',
        '--cfg', str(C.config_path(config_name)),
        '--repeat', '1',
        'seed', str(C.SEED),
        'out_dir', str(out_dir),
        'dataset.split_mode', C.split_mode(),
        'dataset.split_index', str(fold),
        'optim.max_epoch', str(C.MAX_EPOCH),
        'wandb.use', 'False',
    ]
    cmd += list(extra_overrides or [])
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=C.REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"Fold {fold} of {config_name} failed (exit {result.returncode})."
        )
    # main.py nests as <out_dir>/<config_basename>/<seed>/
    return out_dir / config_name / str(C.SEED)


def fold_val_scores(run_dir, metrics=(C.METRIC_PRIMARY, C.METRIC_TIEBREAK),
                    agg='argmax'):
    """Best-validation score per metric for one finished fold.

    Each metric is maximised (or minimised) over epochs independently — the
    same convention `scripts/rank_configs_by_val.py` uses, so CV numbers and
    single-split numbers are computed identically and stay comparable.
    """
    if agg not in _AGG:
        raise ValueError(f"Unsupported metric_agg {agg!r}.")
    selector = _AGG[agg]
    stats_path = run_dir / 'val' / 'stats.json'
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Missing {stats_path}. The fold must have run under "
            f"`train.mode: dgt`."
        )
    records = read_jsonl(stats_path)
    if not records:
        raise RuntimeError(f"{stats_path} is empty.")
    out = {}
    for m in metrics:
        if m not in records[0]:
            raise KeyError(
                f"Metric {m!r} not in {stats_path} "
                f"(available: {sorted(records[0])})."
            )
        best = selector(records, key=lambda r: r[m])
        out[m] = float(best[m])
        out[f'{m}_epoch'] = int(best['epoch'])
    return out


def summarise(per_fold):
    """Aggregate a list of per-fold score dicts into CV mean ± std.

    Std is population std over folds, matching how `agg_runs` and
    `rank_configs_by_val.py` report dispersion over seeds. Note the
    interpretation differs: over folds this is variation across *data splits*,
    which is what §2 asks to be reported, not optimisation noise.
    """
    out = {'n_folds': len(per_fold)}
    for m in (C.METRIC_PRIMARY, C.METRIC_TIEBREAK):
        vals = [f[m] for f in per_fold]
        out[f'{m}_mean'] = statistics.mean(vals)
        out[f'{m}_std'] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out[f'{m}_per_fold'] = vals
    out['best_epoch_per_fold'] = [f[f'{C.METRIC_PRIMARY}_epoch']
                                  for f in per_fold]
    return out


def select_winner(summaries):
    """Apply the §2 rule: primary metric, tiebreak when inside the fold std.

    `summaries` maps config name -> summarise() output. Returns
    (winner, tie_set, used_tiebreak).
    """
    prim, tie = C.METRIC_PRIMARY, C.METRIC_TIEBREAK
    leader = max(summaries, key=lambda c: summaries[c][f'{prim}_mean'])
    lead_mean = summaries[leader][f'{prim}_mean']
    lead_std = summaries[leader][f'{prim}_std']

    tie_set = [
        c for c, s in summaries.items()
        if lead_mean - s[f'{prim}_mean'] <= max(lead_std, s[f'{prim}_std'])
    ]
    if len(tie_set) == 1:
        return leader, tie_set, False
    winner = max(tie_set, key=lambda c: summaries[c][f'{tie}_mean'])
    return winner, tie_set, True
