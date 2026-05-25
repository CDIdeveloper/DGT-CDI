# Session state

> Living "you-are-here" doc. Updated at the end of each session before context auto-compacts.
> Durable docs ([overview.md](overview.md), [tech.md](tech.md), [modeling_routine.md](modeling_routine.md), [trained_models.md](trained_models.md), [graph_transformer.md](graph_transformer.md)) describe *how* the project works. This doc captures *where it is right now* — recent decisions, gotchas encountered, open questions, and the next concrete actions.

**Last updated:** 2026-05-25 (end of session)

---

## Where we are

- **Phase 0 is closed.** Every checkbox in [overview.md → Roadmap → Phase 0](overview.md#phase-0--environment--sanity-check) is `[X]` except two new follow-up items added at the end of the session (regression support in `predict.py`, and a `tests/test_predict.py`) and the original "discuss if following two steps is still necessary" item.
- 2-seed BBBP-DGT-Pipeline run results (unchanged from previous session):

  | Seed | Test AUC |
  |---|---|
  | 0 | 0.7037 |
  | 1 | 0.6653 |
  | **median** | **0.6845** — seed 1 chosen by `retrain_on_trainval.py` |

- New code this session:
  - [scripts/predict.py](../scripts/predict.py) — binary-classification inference on new SMILES. Cuda-only. Input CSV with a SMILES column; output CSV preserves input columns and appends `y_pred_score`, `y_pred_label`, `remarks`. Featurisation copies `torch_geometric.datasets.MoleculeNet`'s atom / bond logic inline (independent of the PyG version's `utils.from_smiles`), then applies the same pre-transform chain `master_loader.py` runs for `PyG-MoleculeNet` (RWSE → SPDE → rings → line-graph → float typecast).
  - [scripts/retrain_on_trainval.py](../scripts/retrain_on_trainval.py) — extended: also bundles the pristine YAML as `<run_dir>/final_model{,_with_test}.config.yaml` and stashes the chosen seed's `best_f1_threshold` (read from `plots/summary.json` if present) into the manifest.
- Documentation updated: [modeling_routine.md](modeling_routine.md) now has **Step 7 — Predict on new data**, plus a "manually-picked seed" deployment note in Step 5 and the deployment-bundle row in the "Where each thing already lives" table.
- **Phases 1-6** (biodegradability + descriptor fusion) not started.

## Immediate next actions

1. **User test on the remote**: run `scripts/predict.py` end-to-end against `results/DGT/BBBP-DGT-Pipeline/final_model.ckpt` to confirm featurisation + checkpoint loading work in the cloud env. The script was authored from static reading of the codebase only (no PyG installed locally on the Mac).
2. Decide on the "discuss if following two steps is still necessary" Roadmap item — the e2e regression test and final verify step. Pending user input.
3. **(optional)** Once `predict.py` is confirmed working, address the two new follow-ups: (a) regression task type and (b) `tests/test_predict.py`.
4. Once Phase 0 is fully closed, **collect Phase 1 inputs from user**: biodegradability CSVs + descriptor list + class balance (see "Open questions" below).

## Recent decisions (this session)

- **Deployment-bundle convention.** A retrained model is now a trio of sibling files in one folder: `final_model{,_with_test}.ckpt`, `final_model{,_with_test}.config.yaml`, `final_model{,_with_test}.json`. `predict.py` finds the bundle via the `<ckpt_stem>.{config.yaml,json}` naming convention, so a deployment is `cp final_model.* deploy/` + `python predict.py --ckpt deploy/final_model.ckpt ...`.
- **Featurisation copied inline, not imported.** PyG 2.0.4's `torch_geometric.utils.from_smiles` is unreliable across versions; replicating `torch_geometric.datasets.MoleculeNet`'s atom / bond featurisation inline guarantees byte-identical features to training and makes `predict.py` PyG-version-tolerant.
- **No `--device` flag.** Cuda-only for now (user hasn't tested CPU). Script fails fast if `torch.cuda.is_available()` is False.
- **Regression support is a separate follow-up.** This round is binary classification only. Output schema for regression will be just `y_pred` + `remarks` (no threshold / label column).

## Recent decisions (carried from previous session)

- **Test methodology.** `dgt` train mode runs the test set **only once**, at the end, on the best-val checkpoint. Per-sample predictions dumped to `<run_dir>/test/predictions.pt`. The upstream `custom` train mode (which evaluates test every epoch) is kept untouched as a regression gate.
- **Two retrain modes for deployment.** `dgt_retrain` (train+val combined, default) and `dgt_retrain_with_test` (train+val+test, **opt-in via `--include-test`**). Both live in [graphgps/train/dgt_retrain.py](../graphgps/train/dgt_retrain.py). Decision pattern: strict by default, permissive on explicit request.
- **Median-seed selection rule** for picking a single deployment model: closest to median test metric (not best, not worst).
- **Documentation homes.** GraphGym registry pattern → [tech.md → Registry pattern](tech.md#registry-pattern-how-components-get-wired-in). Day-to-day workflow → [modeling_routine.md](modeling_routine.md). Model registry → [trained_models.md](trained_models.md). Per-session state → this file.

## Known repo gotchas (encountered & fixed)

Pre-existing bugs in the committed fork. Worth knowing so a fresh checkout doesn't get re-stuck.

1. **`chiro3d_molecule_net.py` → renamed to `chiral3d_molecule_net.py`** to match `master_loader.py:18` and the class name `Chiral3DMoleculeNet`. Renamed via `git mv` on the local; user applied the same rename on the remote.
2. **`graphgps/__init__.py` imports `.pooling` and `.stage`** which don't exist in the fork. User commented both lines out on the remote. The local copy also needs these lines commented for a fresh checkout to run; if/when committed, both clones agree.
3. **`main.py:156` hardcodes `cuda:0`** — correct on cloud CUDA. Would break on CPU/Mac, but we run cloud-only.
4. **Dumped `<run_dir>/config.yaml` can't be re-loaded by yacs** (strict mode rejects runtime-set keys `run_dir` / `params` / `run_id`). Workaround in `retrain_on_trainval.py`: auto-find `configs/**/<run_dir_name>.yaml` and use the pristine original; `--orig-config <path>` overrides if auto-search fails. `predict.py` reuses the same convention; deployment bundle ships a pristine copy alongside the ckpt to make this transparent.

## Environment quirks (user's remote `/home/jovyan/`)

- **Two conda installations on the box**: `/opt/conda` (system, `conda` on PATH) and `/home/jovyan/miniforge3` (where the `dgt` env actually lives, created via mamba). `conda activate dgt` by name **does not work** because `dgt` isn't in conda's `envs_dirs`. Use **`mamba activate dgt`** (works), or `conda activate /home/jovyan/miniforge3/envs/dgt` (explicit path).
- **libgomp `GOMP_5.0` clash** between pip-torch (old bundled `libgomp`) and conda-forge graph-tool. Resolved by `$CONDA_PREFIX/etc/conda/activate.d/zz-libgomp.sh` which `LD_PRELOAD`s the conda libgomp on every env activation. **Per-env** — if the `dgt` env is recreated, this script needs to be re-installed (commands are in [overview.md Phase 0](overview.md#phase-0--environment--sanity-check)).

## Pip dependencies added beyond the original readme

Missing from `readme.md`'s install list but required for the pipeline. Already pinned in [environment.yaml](../environment.yaml):

- `yacs==0.1.8` — required by GraphGym for the global cfg object.
- `networkx` — imported by `get_rings()` in `transforms.py` next to `graph_tool`.
- `matplotlib==3.8.4`, `scikit-learn==1.4.2` — required by `scripts/analyze_run.py`.
- `pandas` — required by `scripts/predict.py` (typically present as a torchmetrics / sklearn transitive dep; pin explicitly if missing).

## Open questions (block Phase 1 onwards)

Need user input before Phase 1 starts:

- **Biodegradability data shape.** Column names of `train.csv` / `test.csv`. Which columns are descriptors vs metadata?
- **Descriptor dimensionality (`desc_dim`).** And whether descriptors are already standardised or the loader should z-score them on the train set.
- **Class balance** of RD/NRD labels. Drives the loss choice in Phase 3 (`cross_entropy` vs `weighted_cross_entropy` vs `focal_loss`).
- **Train/test split source.** Canonical (from a published benchmark) or your own curated split?
- **Whether the biodeg `datasets/biodegradability/` data should be tracked in git** (a `.gitignore` decision deferred from earlier — flagged in modeling_routine.md indirectly).

---

## Where to start the next session

> **First action:** ask the user
>   1. did `scripts/predict.py` work end-to-end on the remote against `results/DGT/BBBP-DGT-Pipeline/final_model.ckpt` (e.g. predicting on a small sample CSV)?
>   2. which next:
>      - resolve the "discuss if following two steps is still necessary" Roadmap item (e2e regression test + final verify), OR
>      - the two `predict.py` follow-ups (regression support + `tests/test_predict.py`), OR
>      - jump to Phase 1 (biodegradability)?
>
> If (1) is *no* — debug `predict.py` on the remote. The most likely failure modes are: (a) `compute_posenc_stats` signature mismatch on this PyG / graphgps version, (b) the inlined MoleculeNet atom/bond feature maps diverging from what BBBP's training-time `processed/` cache used, or (c) `model.load_state_dict` shape mismatch (would indicate `cfg.share.dim_in` / `dim_out` not being derived correctly without `create_loader`).
>
> If (1) is *yes* and the user chooses **Phase 1**: prompt for the biodeg CSV format, descriptor list, and class balance *before* touching any code. See "Open questions" above for the specific list.
