"""Retrain on train+val combined — automatic median-seed selection (on val).

Usage:
    python scripts/retrain_on_trainval.py <run_dir>

`<run_dir>` is the parent results directory of a finished `train.mode: dgt`
run, e.g. `results/DGT/BBBP-DGT-Pipeline/`.

What this script does:
  1. Reads each seed's `<seed>/val/stats.json` and identifies the seed whose
     best-**validation** metric (cfg.metric_best) is closest to the median
     across seeds — i.e. neither best (cherry-pick) nor worst.
  2. Takes that seed's best-validation epoch from the same record.
  3. Subprocesses `main.py` with overrides — `seed=<chosen>`,
     `optim.max_epoch=<best_epoch+1>`, `train.mode=dgt_retrain` — to retrain
     on **train + val combined** for `best_epoch + 1` epochs.
  4. Writes the **deployment bundle** to `<run_dir>/`:
       - `final_model{,_with_test}.ckpt`         — the retrained checkpoint
       - `final_model{,_with_test}.config.yaml`  — pristine YAML (yacs-reloadable)
       - `final_model{,_with_test}.json`         — manifest (seed metrics,
                                                   best_epoch, best_f1_threshold)
     These three files together are everything `scripts/predict.py` needs to
     run inference on a different server.

Neither the test *split* nor any test *metric* is read in the default mode:
seed choice, epoch budget and training data are all validation-derived, so the
deployment artifact carries no information from the held-out set. Use the
original dgt-mode aggregated mean ± std as the reported generalisation estimate.

ONE EXCEPTION: `best_f1_threshold`, copied into the manifest from the chosen
seed's `plots/summary.json`, is swept over TEST predictions by
`scripts/analyze_run.py`. Metrics at the default 0.5 threshold are unaffected;
only `predict.py --threshold optimal-f1` consumes it. See
documents/projects/paper.md §9 item 4.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# metric_agg -> selector for "best validation epoch"
_AGG = {'argmax': max, 'argmin': min}


def _read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _find_seed_dirs(run_dir: Path):
    """Return per-seed subdirs sorted by seed number."""
    seeds = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not seeds:
        raise RuntimeError(
            f"No per-seed subdirs (integer-named) found under {run_dir}."
        )
    return seeds


def _find_original_config(run_dir: Path) -> Path:
    """Locate the pristine YAML in `configs/` that matches this run_dir.

    The dumped `<run_dir>/config.yaml` contains runtime-set keys (`run_dir`,
    `params`, `run_id`, ...) that yacs rejects when reloaded via
    `cfg.merge_from_file` in strict mode. So we re-feed the original file
    from `configs/`, which has only the user-set fields.
    """
    config_name = run_dir.name  # e.g. 'BBBP-DGT-Pipeline'
    configs_root = REPO_ROOT / 'configs'
    candidates = list(configs_root.rglob(f'{config_name}.yaml'))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find original '{config_name}.yaml' under "
            f"{configs_root}/. Pass --orig-config <path> to point at it "
            "explicitly. (The dumped <run_dir>/config.yaml can't be reused "
            "directly because yacs rejects runtime-set keys on reload.)"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple matches for '{config_name}.yaml': "
            f"{[str(p) for p in candidates]}. Pass --orig-config <path> to "
            "specify which one to use."
        )
    return candidates[0]


def _pick_median_seed(run_dir: Path, metric: str, agg: str):
    """Pick the median-scoring seed on VALIDATION.

    Returns (chosen_seed, chosen_metric, per_seed_dict, median, per_seed_epoch).

    Selection is on validation, never test: the deployment artifact must not be
    chosen using information from the held-out set (dgt_porting_guide.md §7
    item 1). Each seed's score is the metric at its best-validation epoch — the
    same quantity `dgt_train.py` uses to select that seed's checkpoint — read
    from `<seed>/val/stats.json`. The seed's best epoch comes from the same
    record, so this function never opens anything under `<seed>/test/`.
    """
    if agg not in _AGG:
        raise ValueError(
            f"Unsupported metric_agg '{agg}' (expected one of {sorted(_AGG)})."
        )
    selector = _AGG[agg]
    seed_dirs = _find_seed_dirs(run_dir)
    per_seed, per_seed_epoch = {}, {}
    for d in seed_dirs:
        stats_path = d / 'val' / 'stats.json'
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Missing {stats_path}. The original run must have used "
                "train.mode: dgt and completed successfully."
            )
        records = _read_jsonl(stats_path)
        if not records:
            raise RuntimeError(f"{stats_path} is empty.")
        if metric not in records[0]:
            raise KeyError(
                f"Metric '{metric}' not found in {stats_path} (available "
                f"keys: {list(records[0].keys())})."
            )
        best = selector(records, key=lambda r: r[metric])
        per_seed[d.name] = float(best[metric])
        per_seed_epoch[d.name] = int(best['epoch'])

    values = sorted(per_seed.values())
    n = len(values)
    median = values[n // 2] if n % 2 \
        else 0.5 * (values[n // 2 - 1] + values[n // 2])
    chosen = min(per_seed, key=lambda s: abs(per_seed[s] - median))
    return chosen, per_seed[chosen], per_seed, median, per_seed_epoch


def _load_descriptor_info(cfg: dict) -> dict:
    """Resolve descriptor_columns + desc_stats (train-split mean/std) for a
    `line_graph_with_desc` model, so they can be embedded in the deployment
    manifest (predict.py is then self-contained — no dataset at inference).

    Locates the dataset's suffix-keyed desc_stats file from the (resolved)
    dumped config + its descriptor-selection spec — mirroring the loader's cache
    keying (graphgps/loader/dataset/_desc_select.py).
    """
    ds = cfg.get('dataset', {})
    pyg_id = ds['format'].split('-', 1)[1]
    ds_root = Path(ds.get('dir', 'datasets')) / pyg_id
    if not ds_root.is_absolute():
        ds_root = REPO_ROOT / ds_root

    # _desc_select only uses hashlib — import the file directly (no heavy
    # graphgps package import).
    sys.path.insert(0, str(REPO_ROOT / 'graphgps' / 'loader' / 'dataset'))
    from _desc_select import select_descriptor_columns, selection_tag

    with open(ds_root / 'raw' / 'manifest.json') as fh:
        all_cols = json.load(fh)['descriptor_columns']
    inc = ds.get('desc_include') or []
    exc = ds.get('desc_exclude') or []
    cols = ds.get('desc_columns') or []
    if inc or exc or cols:
        selected = select_descriptor_columns(all_cols, inc, exc, cols)
        suffix = '_' + selection_tag(selected)
    else:
        suffix = ''
    stats_path = ds_root / 'processed' / f'desc_stats{suffix}.json'
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"line_graph_with_desc model but descriptor stats not found at "
            f"{stats_path}. The training run must have used "
            f"dataset.standardize_desc: True. Cannot embed descriptor info into "
            f"the deployment manifest."
        )
    with open(stats_path) as fh:
        stats = json.load(fh)
    return {
        'descriptor_columns': stats['descriptor_columns'],
        'desc_dim': stats['desc_dim'],
        'desc_stats': {'mean': stats['mean'], 'std': stats['std']},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Retrain on train+val (or train+val+test) combined using "
                    "the median-test seed's hyperparameters and best-val "
                    "epoch."
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Parent results directory of a finished `train.mode: dgt` run "
             "(e.g. results/DGT/BBBP-DGT-Pipeline/).",
    )
    parser.add_argument(
        "--include-test", action="store_true",
        help="Also include the test split in the retrain (train+val+test "
             "combined). Default: train+val only. CAVEAT: the resulting model "
             "has NO held-out data left. Use only for deployment models where "
             "no further test estimate is needed; re-testing on this dataset's "
             "test split afterwards would be leakage.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the retrain budget. Default: the chosen seed's "
             "best-validation epoch + 1. Pass the CV median best epoch + 1 "
             "when the configuration was selected by cross-validation — that "
             "budget is averaged over folds rather than taken from one seed's "
             "validation curve.",
    )
    parser.add_argument(
        "--orig-config", type=Path, default=None,
        help="Explicit path to the original (pristine) YAML used for this "
             "run. Defaults to auto-discovery: configs/**/<run_dir_name>.yaml. "
             "Specify this if the auto-search can't find a unique match.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")

    dumped_cfg_path = run_dir / "config.yaml"
    if not dumped_cfg_path.is_file():
        raise FileNotFoundError(f"Missing {dumped_cfg_path}")
    with open(dumped_cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    metric_best = cfg.get('metric_best', 'auc')
    metric_agg = cfg.get('metric_agg', 'argmax')

    # main.py reloads via yacs which rejects runtime-set keys in the dumped
    # config; point it at the pristine original from configs/ instead.
    orig_cfg_path = (args.orig_config.resolve()
                     if args.orig_config is not None
                     else _find_original_config(run_dir))
    if not orig_cfg_path.is_file():
        raise FileNotFoundError(f"Missing original config: {orig_cfg_path}")
    print(f"Original config (used for retrain): {orig_cfg_path}")

    # 1. Pick the median seed — on VALIDATION, never test.
    chosen, chosen_metric, per_seed, median, per_seed_epoch = _pick_median_seed(
        run_dir, metric_best, metric_agg
    )
    print(f"Per-seed VAL {metric_best}: {per_seed}")
    print(f"Median val {metric_best}: {median:.4f}")
    print(f"Chosen seed: {chosen} (val {metric_best}={chosen_metric:.4f}, "
          f"closest to median). Test set was not read.")

    # 2. That seed's best-validation epoch sets the retrain budget, unless
    #    overridden (e.g. with a CV-derived budget).
    best_epoch = per_seed_epoch[chosen]
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError(f"--epochs must be >= 1, got {args.epochs}.")
        retrain_epochs = args.epochs
        print(f"Median seed's best-val epoch={best_epoch}; overridden by "
              f"--epochs: retraining for {retrain_epochs} epochs on train+val "
              f"combined.")
    else:
        retrain_epochs = best_epoch + 1
        print(f"Median seed's best-val epoch={best_epoch}; will retrain for "
              f"{retrain_epochs} epochs on train+val combined.")

    # 3. Subprocess main.py with overrides.
    train_mode = "dgt_retrain_with_test" if args.include_test else "dgt_retrain"
    final_out_dir = run_dir / ("final_with_test" if args.include_test else "final")
    if args.include_test:
        print("WARNING: --include-test was set. The retrained model will be "
              "trained on train + val + TEST combined. The original test "
              "metric remains the reported number; do NOT re-evaluate this "
              "model on the same test split (would be leakage).")
    cmd = [
        sys.executable, "main.py",
        "--cfg", str(orig_cfg_path),
        "--repeat", "1",
        "seed", str(int(chosen)),
        "optim.max_epoch", str(retrain_epochs),
        "train.mode", train_mode,
        "out_dir", str(final_out_dir),
        "wandb.use", "False",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # 4. Identify and copy the final ckpt to a stable filename.
    config_name = orig_cfg_path.stem
    final_run_dir = final_out_dir / config_name / str(int(chosen))
    ckpts = sorted((final_run_dir / 'ckpt').glob('*.ckpt'))
    if not ckpts:
        print(f"Warning: no ckpt produced under {final_run_dir / 'ckpt'}.")
        sys.exit(1)
    src_ckpt = ckpts[-1]
    suffix = '_with_test' if args.include_test else ''
    dst_ckpt = run_dir / f'final_model{suffix}.ckpt'
    shutil.copy2(src_ckpt, dst_ckpt)

    # 4a. Bundle the pristine config next to the ckpt so the trio
    # (ckpt + config + manifest) is a self-contained deployment artifact.
    dst_cfg = run_dir / f'final_model{suffix}.config.yaml'
    shutil.copy2(orig_cfg_path, dst_cfg)

    # 4b. If the chosen seed already has plots/summary.json (i.e. the user ran
    # analyze_run.py), pull `best_f1_threshold` so predict.py can use the
    # 'optimal-f1' threshold without needing the seed's plots/ folder shipped
    # alongside.
    chosen_summary = run_dir / chosen / 'plots' / 'summary.json'
    best_f1_threshold = None
    if chosen_summary.is_file():
        try:
            with open(chosen_summary) as fh:
                best_f1_threshold = json.load(fh).get('best_f1_threshold')
        except Exception as e:
            print(f"Warning: could not read 'best_f1_threshold' from "
                  f"{chosen_summary}: {e}")
    else:
        print(f"Note: {chosen_summary} not found; skipping "
              "'best_f1_threshold' in manifest. Run scripts/analyze_run.py "
              f"on results/.../{chosen}/ before retraining if you want "
              "predict.py --threshold optimal-f1 to work.")

    # 4c. For descriptor-fusion models, embed descriptor_columns + desc_stats so
    # predict.py is self-contained at inference (no dataset needed).
    descriptor_info = {}
    if cfg.get('gnn', {}).get('head') == 'line_graph_with_desc':
        descriptor_info = _load_descriptor_info(cfg)
        print(f"Embedding descriptor spec into manifest: "
              f"{descriptor_info['desc_dim']} columns.")

    # 5. Write a tiny manifest for downstream consumers.
    manifest = {
        'source_run': str(run_dir),
        'config_dumped': str(dumped_cfg_path),
        'config_original': str(orig_cfg_path),
        'config_name': config_name,
        'metric_best': metric_best,
        'seed_selected_on': 'validation',
        'per_seed_val_metric': per_seed,
        'median_val_metric': median,
        'chosen_seed': int(chosen),
        'chosen_seed_val_metric': chosen_metric,
        'best_epoch_on_original_val_split': best_epoch,
        'retrain_epochs_overridden': args.epochs is not None,
        'best_f1_threshold': best_f1_threshold,
        'retrain_epochs': retrain_epochs,
        'train_mode': train_mode,
        'included_test_in_training': bool(args.include_test),
        'final_run_dir': str(final_run_dir),
        'final_ckpt': str(src_ckpt),
        'final_model_copy': str(dst_ckpt),
        'final_model_config': str(dst_cfg),
        'note': (
            'Trained on train+val+TEST combined; no held-out data remains.'
            if args.include_test else
            'Trained on train+val combined; test set was held out.'
        ),
    }
    manifest.update(descriptor_info)
    manifest_path = run_dir / f'final_model{suffix}.json'
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print()
    print("DONE.")
    print(f"  Final ckpt:        {src_ckpt}")
    print(f"  Convenience copy:  {dst_ckpt}")
    print(f"  Bundled config:    {dst_cfg}")
    print(f"  Manifest:          {manifest_path}")


if __name__ == '__main__':
    main()
