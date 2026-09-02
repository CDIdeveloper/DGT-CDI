"""Stratified 5-fold cross-validation over the TRAIN split, for every config.

The leak-free selection procedure of documents/dgt_porting_guide.md §2,
replacing the single 90/10 validation split used for the first ablation. Each
fold trains a fresh model; the held-out fold is validation; the fixed 278-row
test split is never touched by any fold.

Why this matters: the single-split ablation could not order its four
configurations — every gap fell inside the seed standard deviation, and
re-training one seed reordered the ranking (documents/projects/paper.md §6.1).
CV evaluates on all 5264 training molecules instead of 526, and reports
dispersion across data splits rather than optimisation noise.

Usage
-----
    python scripts/cv/run_cv.py                    # all configs, all folds
    python scripts/cv/run_cv.py --configs A B      # a subset
    python scripts/cv/run_cv.py --folds 0 1        # a subset of folds
    python scripts/cv/run_cv.py --dry-run          # print the plan, run nothing

Resumable: a finished (config, fold) cell is recorded in dgt_cv_results.json
and skipped on re-run, so a crash or Ctrl-C costs at most one fold. Pass
--force to recompute regardless.

Cost: len(configs) x K_FOLDS training runs. For the 4-config biodeg ablation at
50 epochs that is 20 runs, roughly 4.3 h on one GPU.

Writes dgt_cv_results.{json,md} under results/DGT_cv/.
"""
import argparse
import json
import sys
from pathlib import Path

# Sibling modules are imported by bare name (mirrors scripts/analyze_run.py's
# handling of _eval_plots), so the directory must be importable regardless of
# where the script is invoked from.
_CV_DIR = Path(__file__).resolve().parent
if str(_CV_DIR) not in sys.path:
    sys.path.insert(0, str(_CV_DIR))

import dgt_cv_config as C  # noqa: E402
import dgt_common as common  # noqa: E402


def _load_results():
    if C.RESULTS_JSON.is_file():
        with open(C.RESULTS_JSON) as fh:
            return json.load(fh)
    return {'dataset': C.DATASET, 'k_folds': C.K_FOLDS,
            'random_state': C.RANDOM_STATE, 'max_epoch': C.MAX_EPOCH,
            'metric_primary': C.METRIC_PRIMARY,
            'metric_tiebreak': C.METRIC_TIEBREAK, 'cells': {}}


def _save_results(results):
    C.RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_JSON, 'w') as fh:
        json.dump(results, fh, indent=2)


def _render_md(results, summaries, winner, tie_set, used_tiebreak):
    prim, tie = C.METRIC_PRIMARY, C.METRIC_TIEBREAK
    order = sorted(summaries, key=lambda c: -summaries[c][f'{prim}_mean'])
    lines = [
        f"# DGT {C.K_FOLDS}-fold CV — `{C.DATASET}`",
        "",
        f"Stratified {C.K_FOLDS}-fold on the TRAIN split "
        f"(`random_state={C.RANDOM_STATE}`), fresh model per fold, "
        f"{C.MAX_EPOCH} epochs, seed {C.SEED}. The fixed 278-row test split is "
        f"untouched by every fold.",
        "",
        f"Selection: **{prim.upper()}** primary, **{tie.upper()}** tiebreak when "
        f"the gap to the leader falls inside the fold std "
        f"(dgt_porting_guide.md §2).",
        "",
        f"| Rank | Config | Folds | CV {prim.upper()} (mean ± std) | "
        f"CV {tie.upper()} (mean ± std) | Best epoch (median) |",
        "|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(order, 1):
        s = summaries[c]
        eps = sorted(s['best_epoch_per_fold'])
        med = eps[len(eps) // 2] if eps else '—'
        mark = ' ←' if c == winner else ''
        lines.append(
            f"| {i} | `{c}`{mark} | {s['n_folds']} | "
            f"{s[f'{prim}_mean']:.4f} ± {s[f'{prim}_std']:.4f} | "
            f"{s[f'{tie}_mean']:.4f} ± {s[f'{tie}_std']:.4f} | {med} |"
        )
    lines += ["", "## Per-fold values", "",
              f"| Config | CV {prim.upper()} per fold | CV {tie.upper()} per fold |",
              "|---|---|---|"]
    for c in order:
        s = summaries[c]
        f1s = ', '.join(f"{v:.4f}" for v in s[f'{prim}_per_fold'])
        aucs = ', '.join(f"{v:.4f}" for v in s[f'{tie}_per_fold'])
        lines.append(f"| `{c}` | {f1s} | {aucs} |")

    lines += ["", "## Verdict", ""]
    if used_tiebreak:
        lines += [
            f"{len(tie_set)} configurations tie on {prim.upper()} within the "
            f"fold std: {', '.join('`' + c + '`' for c in tie_set)}.",
            "",
            f"Broken on {tie.upper()} → **`{winner}`**.",
        ]
    else:
        lines.append(f"**`{winner}`** leads on {prim.upper()} outright — no "
                     f"tiebreak needed.")
    lines += [
        "",
        "Record this verdict before reading any test number. The test split "
        "is scored once, for this configuration only, by `run_final.py`.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--configs', nargs='+', default=C.CONFIGS,
                    help="Config basenames to evaluate. Default: all in "
                         "dgt_cv_config.CONFIGS.")
    ap.add_argument('--folds', nargs='+', type=int,
                    default=list(range(C.K_FOLDS)),
                    help=f"Folds to run. Default: 0..{C.K_FOLDS - 1}.")
    ap.add_argument('--force', action='store_true',
                    help="Recompute cells already present in the results file.")
    ap.add_argument('--dry-run', action='store_true',
                    help="Print the plan and exit without training.")
    args = ap.parse_args()

    for f in args.folds:
        if not 0 <= f < C.K_FOLDS:
            ap.error(f"--folds value {f} out of range 0..{C.K_FOLDS - 1}")

    results = _load_results()
    todo = [(c, f) for c in args.configs for f in args.folds
            if args.force or f"{c}|{f}" not in results['cells']]
    done = len(args.configs) * len(args.folds) - len(todo)

    print(f"{C.DATASET}: {len(args.configs)} configs x {len(args.folds)} folds "
          f"= {len(args.configs) * len(args.folds)} cells "
          f"({done} already done, {len(todo)} to run).")
    if args.dry_run:
        for c, f in todo:
            print(f"  would run: {c} fold {f}")
        return

    for n, (config_name, fold) in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {config_name} — fold {fold}", flush=True)
        run_dir = common.run_fold(config_name, fold)
        scores = common.fold_val_scores(run_dir)
        results['cells'][f"{config_name}|{fold}"] = {
            'config': config_name, 'fold': fold,
            'run_dir': str(run_dir), **scores,
        }
        _save_results(results)  # checkpoint after EVERY cell
        print(f"  {C.METRIC_PRIMARY}={scores[C.METRIC_PRIMARY]:.4f} "
              f"{C.METRIC_TIEBREAK}={scores[C.METRIC_TIEBREAK]:.4f}", flush=True)

    # Summarise whatever is complete.
    summaries = {}
    for c in args.configs:
        cells = [v for k, v in results['cells'].items() if v['config'] == c]
        if len(cells) != C.K_FOLDS:
            print(f"NOTE: {c} has {len(cells)}/{C.K_FOLDS} folds — excluded "
                  f"from the ranking.", file=sys.stderr)
            continue
        summaries[c] = common.summarise(sorted(cells, key=lambda v: v['fold']))
    if not summaries:
        print("No config has a complete fold set yet; nothing to rank.")
        _save_results(results)
        return

    winner, tie_set, used_tiebreak = common.select_winner(summaries)
    results['summaries'] = summaries
    results['winner'] = winner
    results['tie_set'] = tie_set
    results['used_tiebreak'] = used_tiebreak
    _save_results(results)

    md = _render_md(results, summaries, winner, tie_set, used_tiebreak)
    C.RESULTS_MD.write_text(md + "\n")
    print("\n" + md)
    print(f"Wrote {C.RESULTS_JSON}\nWrote {C.RESULTS_MD}")


if __name__ == '__main__':
    main()
