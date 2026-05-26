# Modeling routine

Practical notes for running the DGT training pipeline day-to-day. For the conceptual model, see [overview.md](overview.md); for the implementation inventory, [tech.md](tech.md).

This doc covers five things:
1. [Where each thing already lives](#where-each-thing-already-lives) — file/location reference for testing, metrics, checkpoints, predictions, plots.
2. [Step-by-step procedure](#step-by-step-procedure) — env setup → training+val → testing → analysis → recording the best model → predicting on new data.
3. [Hyperparameter exploration](#hyperparameter-exploration) — which knobs matter most, with typical sweep ranges.
4. [Recommended workflow](#recommended-workflow-cleanup-between-runs) — what to clean between runs and why (situation-based + by parameter category).
5. (External) [trained_models.md](trained_models.md) — registry where the best model from each run is documented.

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
| **Deployment bundle** (ckpt + config + manifest) | Written by `scripts/retrain_on_trainval.py` | `results/DGT/<config_name>/final_model{,_with_test}.{ckpt,config.yaml,json}` |
| **Predictions on new SMILES** | Run `scripts/predict.py` | output CSV at the path you pass via `--output-csv` |
| W&B live metrics (opt-in) | Yes (if `wandb.use: True`) | the W&B dashboard for the configured project |

**Two important nuances:**

- The upstream `train.mode: custom` runs **test every epoch** and never persists per-sample predictions — there are no plots.
- `train.mode: dgt` (this fork's parallel alternative) **only touches test once**, on the best-val checkpoint, and dumps `predictions.pt`. Plots are then built post-hoc by `scripts/analyze_run.py`.

---

## Step-by-step procedure

A clean run from a fresh shell on the remote, using BBBP as the worked example. Substitute your own config / dataset / `<run_dir>` paths as needed.

### Step 1 — Set up the environment

**Pipeline at a glance.** With `--repeat N seed 0`, training runs as `N` independent seeds (`0..N-1`). Inside each seed: training + validation runs every epoch; the test set is **held out** and run **once** at the very end on that seed's best-val checkpoint, producing `<seed>/test/stats.json` (one line) and `<seed>/test/predictions.pt` — see Step 3 for the mechanics. After all `N` seeds finish, `agg_runs()` in `main.py` reads each seed's `train/val/test/stats.json` and writes **mean ± std across seeds** into `results/DGT/<config_name>/agg/` — that folder is the run's headline result; Step 5 covers how to read it.

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
# --repeat 4 → seeds 0, 1, 2, 3 (seeds starting from 0 different seed for each repeat).
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

#### Picking which seed's checkpoint to keep for downstream use

The **aggregated** mean ± std (from `agg/test/best.json`) is the right number to **report** — it captures uncertainty honestly. But if you need a single concrete model for deployment, predicting on new molecules, or sharing a checkpoint, you have to pick **one** seed's `.ckpt` from the `--repeat N` runs.

The honest selection rule is **pick the seed whose test metric is closest to the median across seeds**:

- ✗ **Don't pick the best seed.** That's cherry-picking — the resulting checkpoint is biased upward relative to the population, so downstream performance will systematically under-perform expectations set by the reported mean.
- ✗ **Don't pick the worst seed.** Equally unrepresentative in the other direction.
- ✓ **Pick the seed closest to the median test metric.** Representative of typical behaviour and closest to the aggregated mean you reported.

Concretely (BBBP example — replace `auc` with `mae` for regression, etc.):

```bash
# Per-seed test metric:
for s in 0 1 2 3; do
  printf "seed %s: " "$s"
  python -c "
import json
last = [json.loads(l) for l in open('results/DGT/BBBP-DGT-Pipeline/$s/test/stats.json')][-1]
print(last['auc'])
"
done
```

Inspect the four numbers, identify the median (for `--repeat 4` that's the average of the two middle values), and pick the seed whose AUC is closest to it. Its checkpoint at `results/DGT/BBBP-DGT-Pipeline/<seed>/ckpt/<best_epoch>.ckpt` is the model to keep / deploy / share — record that path in [trained_models.md](trained_models.md) in the next step.

> **Deploying a manually-picked seed.** If you skip the retrain step and just want to deploy the chosen seed's `.ckpt` to another server, also copy the **pristine** YAML from `configs/` next to it — e.g. `cp configs/physiology/BBBP-DGT-Pipeline.yaml <deploy_dir>/`. The dumped `results/DGT/<config_name>/config.yaml` is **not** reloadable (yacs rejects its runtime-set keys). `scripts/predict.py` then needs `--orig-config <deploy_dir>/BBBP-DGT-Pipeline.yaml` since the auto-discovery convention only finds bundles named `<ckpt_stem>.config.yaml`.

**Optional alternative — retrain for deployment.** After picking the median seed from the per-seed runs, you can fold val into the training set and retrain on more data for the median seed's best-val epoch. The result is a single deterministic model trained on more data, suitable for deployment / serving / predicting on new molecules.

#### Two senses of "test data is used"

Before invoking the retrain, it's worth being explicit about what "use test data" means — these are two different things and they're often conflated:

1. **Test labels enter training** — test data is concatenated into the training set; the model's weights are updated against test labels. Increases the model's effective training-set size but **destroys** the held-out test set forever.
2. **Test set is evaluated on the trained model** — predictions are run on test, a metric is computed. This is what the original `dgt`-mode runs already did.

The default retrain uses **(no test labels in training, no re-evaluation on test)** — strictest methodology. An opt-in `--include-test` flag is available for the case described below.

#### Default — train+val only (recommended)

```bash
python scripts/retrain_on_trainval.py results/DGT/BBBP-DGT-Pipeline/
```

- Training set: `train + val` combined.
- The test set's labels never enter the loss; test is not re-evaluated either. The retrained model has no held-out test estimate of its own; the **original dgt-mode aggregated mean ± std remains the reported generalisation estimate** (it's an estimate of a close-relative model trained on just `train`).

#### Opt-in — train+val+test combined (`--include-test`)

```bash
python scripts/retrain_on_trainval.py results/DGT/BBBP-DGT-Pipeline/ --include-test
```

- Training set: `train + val + test` combined.
- Use case: a deployment-only model where you want to squeeze every available label into training, and you'll never need to re-test on this dataset's test split. Defensible because the original dgt-mode runs already produced and reported a valid test estimate.
- Caveats — read carefully:
  - The model is trained on **every available label**. No held-out data remains.
  - The original test metric is a **lower-bound proxy** for this model (the retrained-with-test model has seen more data than the dgt-mode model whose test metric you reported, so likely does at least as well; you can't verify without a brand-new held-out set).
  - **Re-evaluating this model on the same test split in the future would be leakage** — its predictions there are not generalisation estimates.

#### What the script does (either mode)

1. Reads each seed's `<seed>/test/stats.json`, identifies the seed whose test metric is closest to the median across seeds (neither cherry-picked best nor worst).
2. Reads that seed's `best_epoch` from `<seed>/test/predictions.pt`.
3. Subprocesses `main.py` with `train.mode: dgt_retrain` (default) or `train.mode: dgt_retrain_with_test` (with `--include-test`), `seed=<chosen>`, `optim.max_epoch=<best_epoch+1>`. Both train modes live in [graphgps/train/dgt_retrain.py](../graphgps/train/dgt_retrain.py).
4. Combines the selected splits into a single training loader, trains for the given budget, saves one checkpoint.

Outputs:
- `<run_dir>/final{,_with_test}/<config_name>/<chosen_seed>/ckpt/<final_epoch>.ckpt` — the retrained model (where main.py writes it; the three `final_model.*` files below are convenience copies sitting at the run root).

The three `final_model{,_with_test}.*` files at the run root form a **self-contained deployment bundle**. Copy the trio to any other server and feed them to `scripts/predict.py` (see [Step 7 — Predict on new data](#step-7--predict-on-new-data)). Each file has a distinct purpose:

| File | What it is | Why all three are needed |
|---|---|---|
| `final_model{,_with_test}.ckpt` | Model weights (the learned parameters from `torch.save`). | Loaded into the model via `load_state_dict()`. Without it, no inference. |
| `final_model{,_with_test}.config.yaml` | Copy of the **pristine** YAML config from `configs/`. | Needed to **build the model architecture** (layer count, hidden dim, encoder stack, head type, …) *before* weights can be loaded. The auto-dumped `<run_dir>/config.yaml` cannot be reused — yacs strict-mode rejects its runtime-set keys (`run_dir`, `params`, `run_id`). |
| `final_model{,_with_test}.json` | Manifest. | Records (a) **provenance** — per-seed test metrics, median, chosen seed, `best_epoch`, retrain budget, `train_mode`, `included_test_in_training` — so the selection rationale is auditable later; (b) `best_f1_threshold` (read from the chosen seed's `plots/summary.json` if present) so `predict.py --threshold optimal-f1` works on a deployment server *without* shipping the seed's `plots/` directory. |

Worth the extra run only if you have a definite deployment target. For reporting / handoff, the median-seed checkpoint from the original dgt run is sufficient — the retrain is an optional "use all the labels for the final model" step.

### Step 6 — Document the best model in [trained_models.md](trained_models.md)

For each run you want to record, add an entry to [trained_models.md](trained_models.md) with:
- Date, config path, repeat count, git SHA (so the result is reproducible).
- Checkpoint path of the chosen seed (e.g. seed with median or best test metric).
- Best-val and test metrics; reference the `plots/summary.json` for full numbers.
- Anything noteworthy (class imbalance, manual stopping, unusual losses, etc.).

Keep the actual `.ckpt` files in `results/`. When you want to deploy or share a model, copy the chosen one out and reference it from the trained_models entry.

### Step 7 — Predict on new data

Use [scripts/predict.py](../scripts/predict.py) to run a trained DGT model on new SMILES. Supports both **binary classification** (`task_type: classification_binary`) and **single-target regression** (`task_type: regression`); the task type is read from the bundled config — no CLI flag needed.

**Inputs.** A CSV with a SMILES column (default name: `smiles`; override via `--smiles-col`). Any other columns — IDs, third-party labels, metadata — are preserved verbatim in the output.

**Two deployment scenarios:**

1. **Same server / inside the repo** — the pristine YAML is auto-discovered from `configs/`:
   ```bash
   python scripts/predict.py \
     --ckpt   results/DGT/BBBP-DGT-Pipeline/final_model.ckpt \
     --smiles-csv  new_molecules.csv \
     --output-csv  predictions.csv
   ```
2. **Different server (the deployment bundle).** Copy the three `final_model{,_with_test}.*` sibling files from `<run_dir>/` to a deploy folder, then:
   ```bash
   python scripts/predict.py \
     --ckpt        deploy/final_model.ckpt \
     --smiles-csv  new_molecules.csv \
     --output-csv  predictions.csv
   ```
   `predict.py` finds the bundled `final_model.config.yaml` and `final_model.json` automatically (sibling-file convention: `<ckpt_stem>.config.yaml` and `<ckpt_stem>.json`).

**Threshold (classification only).** Default 0.5 for `y_pred_label`. To use the F1-optimal threshold learned during analysis, pass `--threshold optimal-f1`; this requires the chosen seed's `best_f1_threshold` to have been recorded in `final_model.json` (automatic when `analyze_run.py` was run before `retrain_on_trainval.py`; otherwise pass a numeric threshold). The flag is silently ignored for regression checkpoints.

**Output CSV.** All input columns preserved + appended columns (schema depends on task type):

*Binary classification:*
- `y_pred_score` — class-1 probability (NaN for invalid SMILES).
- `y_pred_label` — 0/1 at the chosen threshold (NaN for invalid SMILES).
- `remarks` — empty on success; reason string on failure (e.g. `invalid SMILES: ...`).

*Regression:*
- `y_pred` — predicted target value (NaN for invalid SMILES).
- `remarks` — empty on success; reason string on failure.

**Manually-picked seed (no retrain).** If you didn't run `retrain_on_trainval.py` and want to predict from a hand-picked seed's `<seed>/ckpt/<best_epoch>.ckpt`, point `--orig-config` at the pristine YAML in `configs/`:
```bash
python scripts/predict.py \
  --ckpt        results/DGT/BBBP-DGT-Pipeline/1/ckpt/41.ckpt \
  --orig-config configs/physiology/BBBP-DGT-Pipeline.yaml \
  --smiles-csv  new_molecules.csv \
  --output-csv  predictions.csv
```

**Cuda-only.** The script fails fast if `torch.cuda.is_available()` returns False — CPU support is not implemented yet.

**Featurisation.** SMILES → atom / bond features inline (matching `torch_geometric.datasets.MoleculeNet` exactly), then the same pre-transform chain `master_loader.py` runs for `PyG-MoleculeNet` (RWSE → SPDE → rings → line-graph → float typecast). No `datasets/` cache is consulted; the script does not require the original training dataset to be present.

---

## Hyperparameter exploration

For **reproducing the paper on benchmark datasets** (BBBP, QM9, etc.), the per-dataset settings in [tech.md → Per-dataset DGT hyperparameters](tech.md#per-dataset-dgt-hyperparameters) already match what the authors landed on — usually no sweep needed.

> **Looking up what a YAML field means** (rather than which to tune): see [config_reference.md](config_reference.md) — line-by-line annotation of `BBBP-DGT-Pipeline.yaml` and `FreeSolv-DGT-Pipeline.yaml`, merged into one reference.

For **your own data** (e.g. biodegradability in Phases 3+), a small grid over the highest-impact knobs is usually worthwhile because the right values depend on dataset size and class imbalance, which neither you nor the paper knows in advance. The table below groups parameters by typical impact on test performance; tune from the top down.

| Tier | Parameter (location) | Default (BBBP-DGT-Pipeline) | Typical sweep |
|---|---|---|---|
| **High impact** | `optim.base_lr` | `4e-4` | `1e-3`, `4e-4`, `1e-4` |
|  | `gnn.dim_inner` + `gt.dim_hidden` (must match) | `128` | `64`, `128`, `256` |
|  | `gt.layers` | `4` | `3`, `4`, `6` |
|  | `optim.weight_decay` | `1e-2` | `1e-1`, `1e-2`, `1e-3`, `0` |
| **Medium impact** | `gt.n_heads` (must divide `dim_hidden`) | `16` | `8`, `16`, `32` |
|  | `gt.dropout`, `gt.attn_dropout` | `0.0`, `0.3` | `0.0–0.2` for `dropout`; `0.1–0.4` for `attn_dropout` |
|  | `gnn.layers_post_mp` | `3` | `2`, `3` |
|  | `train.batch_size` | `32` | `16`, `32`, `64` (memory-bound) |
|  | `optim.num_warmup_epochs` | `10` | ~10 % of `max_epoch` |
| **Pre-transform** (requires cleanup — see below) | `dataset.spd_max_length` | `8` | `6`, `8`, `12` |
|  | `dataset.rings_max_length` | `18` | `6`, `12`, `18` |
|  | `posenc_RWSE.kernel.times_func` | `range(1,17)` | `range(1,9)`, `range(1,17)`, `range(1,25)` |

**One config per variant.** When sweeping, copy the YAML to a new filename rather than editing in place:

```
configs/physiology/BBBP-DGT-Pipeline.yaml            # baseline
configs/physiology/BBBP-DGT-Pipeline-lr1e3.yaml      # base_lr = 1e-3
configs/physiology/BBBP-DGT-Pipeline-dim256-L6.yaml  # dim_inner=256, layers=6
```

Each variant lands in its own `results/DGT/<config_name>/` directory, so seeds don't collide, `agg_runs()` doesn't mix variants, and the "which config produced which numbers" trail stays clean.

---

## Recommended workflow (cleanup between runs)

When and what to clean between training runs. Default is to clean **nothing** — the file caches and per-seed run dirs are reused or refreshed automatically. Manual cleanup is only needed in the specific situations below.

| Situation | Action | Reason |
|---|---|---|
| Re-running the **same config** | Nothing | PyG's `datasets/<DatasetName>/processed/` cache is deterministic given the same code + pre-transform params, so reusing it saves the (sometimes minutes-long) RWSE / rings / SPD / line-graph caching step. Per-seed run directories under `results/DGT/<config_name>/<seed>/` are wiped and recreated automatically by `makedirs_rm_exist` in [main.py:96](../main.py#L96). Manual cleaning would just slow the iteration loop. |
| Changed pre-transform params (`spd_max_length`, `rings_max_length`, RWSE `times_func`, etc.) | `rm -rf datasets/<DatasetName>/processed/` | PyG keys the cache by **dataset class + root path only** — it does **not** detect changes to your YAML's pre-transform parameters. Without manual cleanup, the run silently reuses stale processed data and you'd be training on the *old* parameter values. Leave `raw/` alone so the dataset isn't re-downloaded. |
| Changed `--repeat N` between runs (e.g. 4 → 2) | `rm -rf results/DGT/<config_name>/` before re-running | Each per-seed dir is wiped on re-run, but the parent dir is not. Going from `--repeat 4` to `--repeat 2` leaves the old `2/` and `3/` seed dirs in place, and `agg_runs()` at the end of `main.py` will fold them into the aggregated mean ± std alongside the new runs — mixing old and new results. |
| Changed dataset loader / featurisation code (`graphgps/loader/dataset/...`) | `rm -rf datasets/<DatasetName>/processed/` | The processed cache reflects the previous loader's output (atom / bond features, edge_index, attached tensors). PyG doesn't notice that the loader changed, so a stale cache would silently feed the model wrong / old data. |

### Cleanup by parameter category (quick reference for HPO sweeps)

If you're only changing YAML hyperparameters (no code changes), this table maps each parameter category to the cleanup action needed. Tiers reference the [Hyperparameter exploration](#hyperparameter-exploration) section above.

| Parameter category | Cleanup needed? | Why |
|---|---|---|
| `optim.*` (lr, weight_decay, max_epoch, scheduler, warmup) | **No** | Optimization-only — no effect on data cache; per-seed run dirs auto-wipe. |
| `gt.*`, `gnn.*` (layers, dim_hidden, n_heads, dropout, post_mp depth, etc.) | **No** | Model architecture only. Dataset cache unaffected. |
| `train.batch_size`, `train.eval_period` | **No** | Training control, not data. |
| `dataset.spd_max_length`, `dataset.rings_max_length`, `posenc_RWSE.kernel.times_func` | **Yes** — `rm -rf datasets/<DatasetName>/processed/` | Pre-transform parameters. PyG's `processed/` cache won't detect the YAML change. |
| `--repeat N` between runs | **Yes** — `rm -rf results/DGT/<config_name>/` | Old seed dirs would otherwise be folded into `agg_runs()`'s mean ± std. |

**Bottom line:** cleanup is only needed for **pre-transform** changes and the `--repeat N` case. All Tier-1 / Tier-2 knobs (lr, dim, layers, dropout, batch size, etc.) require **zero cleanup**.
