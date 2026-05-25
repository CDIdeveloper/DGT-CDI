# Modeling routine

Practical notes for running the DGT training pipeline day-to-day. For the conceptual model, see [overview.md](overview.md); for the implementation inventory, [tech.md](tech.md).

This doc covers four things:
1. [Where each thing already lives](#where-each-thing-already-lives) — file/location reference for testing, metrics, checkpoints, predictions, plots.
2. [Step-by-step procedure](#step-by-step-procedure) — env setup → training+val → testing → analysis → recording the best model.
3. [Recommended workflow](#recommended-workflow-cleanup-between-runs) — what to clean between runs and why.
4. (External) [trained_models.md](trained_models.md) — registry where the best model from each run is documented.

---

## Where each thing already lives

Default path conventions when running `python main.py --cfg <cfg> ...`. Substitute `<config_name>` for the config's basename (e.g. `BBBP-DGT-Pipeline`) and `<seed>` for the per-seed subdirectory (e.g. `0`, `1`, ...).

| Concern | Already done? | Location |
|---|---|---|
| Forward pass on test set | Yes | `eval_epoch()` / `_eval_and_collect()` |
| Binary-classification metrics (ROC-AUC, F1, precision, recall, accuracy) | Yes | `CustomLogger.classification_binary()` in [graphgps/logger.py](../graphgps/logger.py) |
| Regression metrics (MAE, RMSE, R², Spearman) | Yes | `CustomLogger.regression()` in [graphgps/logger.py](../graphgps/logger.py) |
| Per-epoch metric records (JSONL — one JSON object per epoch) | Yes | `results/DGT/<config_name>/<seed>/{train,val,test}/stats.json` |
| Multi-seed aggregation | Yes | `agg_runs(cfg.out_dir, cfg.metric_best)` at [main.py:202](../main.py#L202) → `results/DGT/<config_name>/agg/` |
| Model checkpoint | Yes (when `train.enable_ckpt: True`) | `results/DGT/<config_name>/<seed>/ckpt/<epoch>.ckpt`. With `ckpt_best: True` + `ckpt_clean: True` (defaults for `train.mode: dgt`) only the **best-val** checkpoint is retained. |
| **Per-sample test predictions** (raw `y_true`, `y_pred_score`) | Only with `train.mode: dgt` | `results/DGT/<config_name>/<seed>/test/predictions.pt` |
| **Plots** (ROC, PR, confusion matrix, scatter, residuals) | Generated post-hoc | `results/DGT/<config_name>/<seed>/plots/*.png` (run `scripts/analyze_run.py`) |
| W&B live metrics (opt-in) | Yes (if `wandb.use: True`) | the W&B dashboard for the configured project |

**Two important nuances:**

- The upstream `train.mode: custom` runs **test every epoch** and never persists per-sample predictions — there are no plots.
- `train.mode: dgt` (this fork's parallel alternative) **only touches test once**, on the best-val checkpoint, and dumps `predictions.pt`. Plots are then built post-hoc by `scripts/analyze_run.py`.

---

## Step-by-step procedure

A clean run from a fresh shell on the remote, using BBBP as the worked example. Substitute your own config / dataset / `<run_dir>` paths as needed.

### Step 1 — Set up the environment

```bash
# On the remote, in the repo root.
mamba activate dgt

# Verify imports and CUDA.
python -c "import torch, torch_geometric, torch_scatter, graph_tool, rdkit; print(torch.cuda.is_available())"
# Expected: True

# Verify libgomp preload (set by activate.d/zz-libgomp.sh).
echo $LD_PRELOAD
# Expected: /home/jovyan/miniforge3/envs/dgt/lib/libgomp.so.1   (or your env's path)
```

If either check fails, see overview.md Phase 0 (env build + LD_PRELOAD section).

### Step 2 — Run training + validation

```bash
# BBBP with the DGT pipeline (train+val each epoch; test held out).
# --repeat 4 → seeds 0, 1, 2, 3 (matches the paper's 4-seed averaging).
python main.py \
  --cfg configs/physiology/BBBP-DGT-Pipeline.yaml \
  --repeat 4 seed 0 wandb.use False
```

What happens:
- Each epoch logs `train/stats.json` and `val/stats.json` under `results/DGT/BBBP-DGT-Pipeline/<seed>/`.
- Whenever a new best `val_auc` is observed, the checkpoint is saved and any older checkpoint is deleted (`ckpt_best=True`, `ckpt_clean=True`).
- The test loader is **not** touched during the loop.
- Console shows `> Epoch K: ... best so far: epoch B val_auc: 0.xx` lines.

### Step 3 — Final test on the best checkpoint (automatic)

After the train+val loop finishes, `dgt_train` automatically:
- Loads the best-val checkpoint.
- Runs the test loader **once**.
- Writes `test/stats.json` (one line, at the best-val epoch).
- Dumps per-sample predictions to `<run_dir>/test/predictions.pt`.

You'll see in the log:
```
[dgt] Loaded checkpoint from epoch B (best by val_auc).
[dgt] Final test stats @ best-val epoch B: {'auc': 0.xx, 'accuracy': ..., ...}
[dgt] Wrote test predictions: .../test/predictions.pt
```

### Step 4 — Build the analysis plots

```bash
# Run for each seed; output goes into <run_dir>/plots/.
python scripts/analyze_run.py results/DGT/BBBP-DGT-Pipeline/0
python scripts/analyze_run.py results/DGT/BBBP-DGT-Pipeline/1
python scripts/analyze_run.py results/DGT/BBBP-DGT-Pipeline/2
python scripts/analyze_run.py results/DGT/BBBP-DGT-Pipeline/3
```

Each invocation creates `<run_dir>/plots/`:
- `roc.png`, `pr.png`, `confusion.png`, `score_hist.png` (for binary classification).
- `scatter.png`, `residual.png`, `residual_hist.png` (for regression).
- `summary.json` — best epoch, best-val metric, test metrics, threshold, confusion matrix.

To view the plots either `rsync` the run dir down to a local machine, or open them via Jupyter / a remote file browser.

### Step 5 — Review results and pick the model to keep

For each seed:

```bash
# Best-val metric, best-test metric, threshold, etc.
cat results/DGT/BBBP-DGT-Pipeline/0/plots/summary.json
```

Across seeds:

```bash
# Mean ± std aggregated by main.py at the end of training.
cat results/DGT/BBBP-DGT-Pipeline/agg/test/best.json   # or agg/val/best.json
```

For binary classification, look at:
- `roc.png` — discriminative power overall.
- `confusion.png` — operating point and class-imbalance behaviour.
- `score_hist.png` — separation between true-0 and true-1 score distributions.

For regression, look at:
- `scatter.png` — quality of fit + outliers.
- `residual.png` — bias as a function of predicted value (look for tilt or fanning).

### Step 6 — Document the best model in [trained_models.md](trained_models.md)

For each run you want to record, add an entry to [trained_models.md](trained_models.md) with:
- Date, config path, repeat count, git SHA (so the result is reproducible).
- Checkpoint path of the chosen seed (e.g. seed with median or best test metric).
- Best-val and test metrics; reference the `plots/summary.json` for full numbers.
- Anything noteworthy (class imbalance, manual stopping, unusual losses, etc.).

Keep the actual `.ckpt` files in `results/`. When you want to deploy or share a model, copy the chosen one out and reference it from the trained_models entry.

---

## Recommended workflow (cleanup between runs)

When and what to clean between training runs. Default is to clean **nothing** — the file caches and per-seed run dirs are reused or refreshed automatically. Manual cleanup is only needed in the specific situations below.

| Situation | Action | Reason |
|---|---|---|
| Re-running the **same config** | Nothing | PyG's `datasets/<DatasetName>/processed/` cache is deterministic given the same code + pre-transform params, so reusing it saves the (sometimes minutes-long) RWSE / rings / SPD / line-graph caching step. Per-seed run directories under `results/DGT/<config_name>/<seed>/` are wiped and recreated automatically by `makedirs_rm_exist` in [main.py:96](../main.py#L96). Manual cleaning would just slow the iteration loop. |
| Changed pre-transform params (`spd_max_length`, `rings_max_length`, RWSE `times_func`, etc.) | `rm -rf datasets/<DatasetName>/processed/` | PyG keys the cache by **dataset class + root path only** — it does **not** detect changes to your YAML's pre-transform parameters. Without manual cleanup, the run silently reuses stale processed data and you'd be training on the *old* parameter values. Leave `raw/` alone so the dataset isn't re-downloaded. |
| Changed `--repeat N` between runs (e.g. 4 → 2) | `rm -rf results/DGT/<config_name>/` before re-running | Each per-seed dir is wiped on re-run, but the parent dir is not. Going from `--repeat 4` to `--repeat 2` leaves the old `2/` and `3/` seed dirs in place, and `agg_runs()` at the end of `main.py` will fold them into the aggregated mean ± std alongside the new runs — mixing old and new results. |
| Changed dataset loader / featurisation code (`graphgps/loader/dataset/...`) | `rm -rf datasets/<DatasetName>/processed/` | The processed cache reflects the previous loader's output (atom / bond features, edge_index, attached tensors). PyG doesn't notice that the loader changed, so a stale cache would silently feed the model wrong / old data. |
