"""Rank finished DGT runs by VALIDATION metric instead of test metric.

Why this exists
---------------
`documents/dgt_porting_guide.md` §2 requires that *all* model choices
(architecture, hyperparameters, feature set, epochs, threshold) are made without
looking at the held-out test set, and §7 item 1 asks you to verify it. The HPO
and descriptor-variant tables in `documents/trained_models.md` and
`documents/projects/gwu.md` were populated from `agg/test/best.json` — i.e. the
configurations were ranked on TEST. This script re-derives that ranking from the
validation split, which every `train.mode: dgt` run already wrote to disk.

No retraining is needed: the artefacts exist. If the val ranking agrees with the
test ranking, the published headline is defensible and you just restate its
basis. If it disagrees, a selection error has been caught.

What "val score" means here
---------------------------
Per seed, the score is the metric at the **best-val epoch** — the same quantity
`dgt_train.py` uses to pick the checkpoint, i.e. a max (or min) over epochs.
That is a *selection score*, not an unbiased generalisation estimate: maximising
over ~50 epochs inflates it. The inflation applies identically to every
configuration, so comparing configurations on it is fair — but do not report
these numbers as generalisation performance. The test set remains the only
estimate of that, and only for the single configuration you select here.

Expected layout (written by `train.mode: dgt`)
----------------------------------------------
    <config_dir>/config.yaml            metric_best + metric_agg
    <config_dir>/<seed>/val/stats.json  JSONL, one record per eval epoch
    <config_dir>/<seed>/test/stats.json JSONL, exactly one final record
    <config_dir>/<seed>/test/predictions.pt   (optional; used to cross-check)

CLI
---
    # rank every config under a results root
    python scripts/rank_configs_by_val.py results/DGT/

    # rank an explicit subset
    python scripts/rank_configs_by_val.py \\
        results/DGT/Biodeg-GWU-DGT-Pipeline \\
        results/DGT/Biodeg-GWU-DGT-Pipeline-WithDesc-nongwu

    # make the selection WITHOUT seeing test (the leak-free way), then commit it
    python scripts/rank_configs_by_val.py results/DGT/ --hide-test

    [--metric auc]        override cfg.metric_best
    [--hide-test]         omit test columns entirely
    [--json  <path>]      machine-readable dump
    [--markdown <path>]   write the table to a file (else stdout)

Reads only. Writes nothing unless --json / --markdown is given.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

# Optional: only needed for the predictions.pt cross-check. The primary path
# (val/stats.json) is pure JSON, so this script runs in a torch-free env.
try:
    import torch
except ImportError:  # pragma: no cover - environment-dependent
    torch = None

# metric_agg -> (higher_is_better, python selector)
_AGG = {'argmax': (True, max), 'argmin': (False, min)}


def _read_jsonl(path):
    """Parse a GraphGym stats.json (one JSON object per line)."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_config(config_dir):
    """Return the dumped config for a config-level run directory.

    GraphGym's `dump_cfg` writes config.yaml to `cfg.out_dir`, which is the
    config-level dir (the parent of the per-seed dirs).
    """
    path = config_dir / 'config.yaml'
    if not path.is_file():
        raise FileNotFoundError(
            f"No config.yaml in {config_dir}. Expected a config-level run dir "
            f"(the parent of the numeric seed dirs), e.g. "
            f"results/DGT/Biodeg-GWU-DGT-Pipeline/."
        )
    with open(path) as fh:
        return yaml.safe_load(fh)


def _seed_dirs(config_dir):
    """Numeric per-seed subdirs, ascending. Skips `agg`, `final`, `plots`, ..."""
    return sorted(
        [d for d in config_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )


def _val_score(seed_dir, metric, agg):
    """Return (best_val_metric, best_epoch) from <seed_dir>/val/stats.json.

    Equivalent to the `best_val_metric` that dgt_train.py stores in
    predictions.pt, but torch-free and still available for runs predating that
    field. Epochs skipped by `eval_period > 1` are simply absent from the file;
    they are carried-forward duplicates in dgt_train and never change the
    extremum.
    """
    path = seed_dir / 'val' / 'stats.json'
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. This run was probably not trained with "
            f"`train.mode: dgt`."
        )
    records = _read_jsonl(path)
    if not records:
        raise RuntimeError(f"{path} is empty.")
    missing = [r for r in records if metric not in r]
    if missing:
        raise KeyError(
            f"Metric '{metric}' not in {path} "
            f"(available: {sorted(missing[0].keys())})."
        )
    _, selector = _AGG[agg]
    best = selector(records, key=lambda r: r[metric])
    return float(best[metric]), int(best['epoch'])


def _test_score(seed_dir, metric):
    """Return the single final test metric, or None if absent."""
    path = seed_dir / 'test' / 'stats.json'
    if not path.is_file():
        return None
    records = _read_jsonl(path)
    if len(records) != 1:
        # dgt mode writes exactly one final test record; anything else means
        # the run used train.mode: custom (test scored every epoch).
        return None
    return float(records[0][metric]) if metric in records[0] else None


def _crosscheck(seed_dir, best_val, tol=1e-6):
    """Compare our val score against predictions.pt's stored best_val_metric.

    Returns a warning string on mismatch, else None. Skipped when torch is
    unavailable or the file predates the stored field.
    """
    path = seed_dir / 'test' / 'predictions.pt'
    if torch is None or not path.is_file():
        return None
    try:
        pred = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:  # pragma: no cover - corrupt/legacy artefact
        return f"{seed_dir}: could not read predictions.pt ({e})"
    stored = pred.get('best_val_metric')
    if stored is None:
        return None
    if abs(float(stored) - best_val) > tol:
        return (f"{seed_dir}: val score {best_val:.6f} disagrees with "
                f"predictions.pt best_val_metric {float(stored):.6f}")
    return None


def _mean_std(values):
    """Mean and POPULATION std — matches GraphGym's agg_runs (np.std, ddof=0),
    so numbers line up with the existing tables in trained_models.md."""
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def _collect(config_dir, metric_override):
    """Build one result row for a config-level run dir."""
    cfg = _load_config(config_dir)
    metric = metric_override or cfg.get('metric_best', 'auc')
    agg = cfg.get('metric_agg', 'argmax')
    if agg not in _AGG:
        raise ValueError(
            f"{config_dir}: unsupported metric_agg '{agg}' "
            f"(expected one of {sorted(_AGG)})."
        )

    seeds = _seed_dirs(config_dir)
    if not seeds:
        return None, []

    warnings = []
    val_scores, test_scores, best_epochs = [], [], []
    for d in seeds:
        val, epoch = _val_score(d, metric, agg)
        val_scores.append(val)
        best_epochs.append(epoch)
        warn = _crosscheck(d, val)
        if warn:
            warnings.append(warn)
        test = _test_score(d, metric)
        if test is not None:
            test_scores.append(test)

    val_mean, val_std = _mean_std(val_scores)
    row = {
        'config': config_dir.name,
        'metric': metric,
        'metric_agg': agg,
        'n_seeds': len(seeds),
        'seeds': [d.name for d in seeds],
        'val_mean': val_mean,
        'val_std': val_std,
        'val_per_seed': val_scores,
        'best_epoch_median': int(statistics.median(best_epochs)),
        'test_mean': None,
        'test_std': None,
        'test_per_seed': test_scores,
    }
    if test_scores:
        row['test_mean'], row['test_std'] = _mean_std(test_scores)
    if len(test_scores) not in (0, len(seeds)):
        warnings.append(
            f"{config_dir.name}: test metric found for only "
            f"{len(test_scores)}/{len(seeds)} seeds; test column is a partial "
            f"aggregate."
        )
    return row, warnings


def _ranked(rows, key, higher_is_better):
    """Return rows that have `key`, sorted best-first."""
    have = [r for r in rows if r[key] is not None]
    return sorted(have, key=lambda r: r[key], reverse=higher_is_better)


def _render(rows, metric, higher_is_better, show_test):
    """Render the ranking as a markdown table plus a verdict."""
    by_val = _ranked(rows, 'val_mean', higher_is_better)
    by_test = _ranked(rows, 'test_mean', higher_is_better)
    test_rank = {r['config']: i + 1 for i, r in enumerate(by_test)}

    m = metric.upper()
    head = ["Rank", "Config", "Seeds", f"Val {m} (mean ± std)", "Best epoch"]
    if show_test:
        head += [f"Test {m} (mean ± std)", "Test rank", "Δ"]

    lines = [
        f"# Configuration ranking by VALIDATION {m}",
        "",
        f"Selection basis: val {m} at each seed's best-val epoch, averaged over "
        f"seeds (population std, matching `agg_runs`).",
        "",
    ]
    if show_test:
        lines += [
            "> Test columns are shown **for comparison only** and must not "
            "drive the choice. `Δ` is (test rank − val rank); non-zero means "
            "the two bases disagree.",
            "",
        ]
    lines += ["| " + " | ".join(head) + " |",
              "|" + "|".join(["---"] * len(head)) + "|"]

    for i, r in enumerate(by_val, start=1):
        cells = [
            str(i),
            r['config'],
            str(r['n_seeds']),
            f"{r['val_mean']:.4f} ± {r['val_std']:.4f}",
            str(r['best_epoch_median']),
        ]
        if show_test:
            if r['test_mean'] is None:
                cells += ["—", "—", "—"]
            else:
                tr = test_rank[r['config']]
                delta = tr - i
                cells += [
                    f"{r['test_mean']:.4f} ± {r['test_std']:.4f}",
                    str(tr),
                    f"{delta:+d}" if delta else "0",
                ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Verdict", ""]
    if not by_val:
        lines.append("No rankable configs found.")
        return "\n".join(lines)

    winner = by_val[0]
    lines.append(
        f"**Val-selected configuration: `{winner['config']}`** "
        f"(val {m} {winner['val_mean']:.4f} ± {winner['val_std']:.4f})."
    )
    if show_test and by_test:
        if by_test[0]['config'] == winner['config']:
            lines += [
                "",
                f"The test-selected winner is the same configuration. The "
                f"published headline is defensible: restate its basis as val "
                f"{m}, and keep the test number as the one-shot estimate.",
            ]
        else:
            lines += [
                "",
                f"⚠️ **Disagreement.** Test would have selected "
                f"`{by_test[0]['config']}` "
                f"(test {m} {by_test[0]['test_mean']:.4f}). The val-selected "
                f"winner above is the leak-free choice; the test score for "
                f"that configuration is its honest one-shot estimate.",
            ]
        moved = [r['config'] for i, r in enumerate(by_val, start=1)
                 if r['test_mean'] is not None
                 and test_rank[r['config']] != i]
        if moved:
            lines += ["", f"Configs whose rank differs between the two bases: "
                          f"{', '.join(moved)}."]
    if not show_test:
        lines += [
            "",
            "Test columns suppressed (`--hide-test`). Commit this selection "
            "before looking at any test number.",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Rank finished DGT runs by validation metric instead of "
                    "test metric (leak-free selection, per "
                    "documents/dgt_porting_guide.md §2).",
    )
    parser.add_argument(
        "paths", type=Path, nargs='+',
        help="Config-level run dirs (e.g. results/DGT/Biodeg-GWU-DGT-Pipeline), "
             "or a results root to scan for them (e.g. results/DGT/).",
    )
    parser.add_argument(
        "--metric", type=str, default=None,
        help="Metric key to rank on. Default: each run's cfg.metric_best.",
    )
    parser.add_argument(
        "--hide-test", action="store_true",
        help="Omit test columns entirely — use this to make and record the "
             "selection before any test number is visible.",
    )
    parser.add_argument(
        "--json", dest="json_out", type=Path, default=None,
        help="Also write the collected rows to this path as JSON.",
    )
    parser.add_argument(
        "--markdown", dest="md_out", type=Path, default=None,
        help="Write the table here instead of stdout.",
    )
    args = parser.parse_args()

    # Resolve inputs to config-level dirs: a path is one if it has numeric seed
    # subdirs, otherwise treat its children as candidates.
    config_dirs = []
    for p in args.paths:
        p = p.resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Not a directory: {p}")
        if _seed_dirs(p):
            config_dirs.append(p)
        else:
            config_dirs.extend(
                sorted(c for c in p.iterdir() if c.is_dir() and _seed_dirs(c))
            )
    config_dirs = sorted(set(config_dirs))
    if not config_dirs:
        raise RuntimeError(
            f"No run directories with numeric seed subdirs found under "
            f"{[str(p) for p in args.paths]}."
        )

    rows, warnings = [], []
    for d in config_dirs:
        try:
            row, warns = _collect(d, args.metric)
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as e:
            warnings.append(f"{d.name}: skipped — {e}")
            continue
        if row is not None:
            rows.append(row)
            warnings.extend(warns)

    if not rows:
        raise RuntimeError("No configs could be ranked; see warnings above.")

    metrics = {r['metric'] for r in rows}
    if len(metrics) > 1:
        raise ValueError(
            f"Configs use different metrics ({sorted(metrics)}); they are not "
            f"comparable. Re-run with --metric to force one."
        )
    metric = rows[0]['metric']
    higher_is_better = _AGG[rows[0]['metric_agg']][0]

    table = _render(rows, metric, higher_is_better, show_test=not args.hide_test)

    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(table + "\n")
        print(f"Wrote {args.md_out}")
    else:
        print(table)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, 'w') as fh:
            json.dump({'metric': metric,
                       'higher_is_better': higher_is_better,
                       'rows': rows}, fh, indent=2)
        print(f"Wrote {args.json_out}")

    if warnings:
        print("\nWarnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)


if __name__ == '__main__':
    main()
