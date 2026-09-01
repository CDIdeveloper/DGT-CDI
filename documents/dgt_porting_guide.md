# Biodeg modelling — workflow & leak-free protocol (porting guide for DGT-CDI)

Purpose: a self-contained handoff so the **DGT (Dual Graph Transformer)** work in repo `DGT-CDI` can
reuse the same dataset, the same leak-free selection protocol, and the same evaluation/reporting
patterns as the sklearn + MPNN work in `trans_learn`, and so the DGT code can be audited for
test-data leakage. Copy this file into `DGT-CDI` and follow §6 (retrofit checklist) and §7 (leakage
audit).

Repo roles: `getrightmol` = data prep; `trans_learn` = sklearn + MPNN (this repo); `DGT-CDI` = DGT;
`biodeg-gwu-cdi` = public release (built later). All three model repos must train/evaluate on the
**same dataset and the same fixed split**, and report the **same metrics** on the **same 278-test**.

---

## 1. Dataset (identical across all model repos)

- **Name / registry:** `biodeg_gwu_no_ind` in `settings.py::DATASET_REGISTRY` (trans_learn).
- **Task:** binary — `1 = readily biodegradable (RB)`, `0 = not readily (NRB)`; inherent (InD) rows removed.
- **Counts:** train **5264** / test **278** (247 descriptor columns). Target col `degradable`, SMILES col `smiles`.
- **Fixed split (D5):** a fixed random train/test split — **the headline; do NOT re-split in DGT.**
  A canonical-SMILES near-duplicate audit found **0 true cross-split duplicates** (the 51
  fingerprint-identical test↔train pairs are stereo/E-Z isomers + aliphatic chain-length homologs,
  a Morgan-radius-2 artifact), so the split is retained as-is. DGT inherits this same split.
- **S3 data:** `s3://cdi-lab-workspaces/ts_project_1/data/biodegradation/GWU/train_test/`
  — `biodeg_gwu_b2_no_ind_{train,test}.parquet`; reusable Morgan fingerprints under `.../fps/`.
- **Feature sets** (column-name suffix selection; MPNN/DGT also have graph-only `none`):
  | name | descriptor columns | count |
  |---|---|---|
  | `none` | (graph only, no descriptors) | 0 |
  | `qm` | endswith `_gwu` | 40 |
  | `rdkit_fg` | endswith `_rdkit`, `_fg` | 207 |
  | `qm_rdkit` | endswith `_gwu`, `_rdkit`, `_fg` | 247 |
- **Data-prep scripts** (in `getrightmol`, copies kept in `trans_learn/documents/project/`):
  `get_biodeg_data_gwu_no_ind.py`, `eda_biodeg_gwu_no_ind.py`, `check_smiles_duplicates_gwu_no_ind.py`.

---

## 2. The leak-free selection protocol  ← the part DGT must replicate

**Invariant:** *all* model choices (architecture, hyperparameters, feature set, epochs, threshold) are
selected by **cross-validation on the TRAIN split only**; the **278-test is scored exactly once**, for
the single final chosen configuration.

- **CV:** stratified **5-fold** on TRAIN. A fresh model per fold, trained from scratch (no warm-start
  carrying information across folds). An inner validation split (10%) is carved **from the training
  folds** for monitoring/early-stopping — never from the test set.
- **Selection metric:** **F1** (primary); **ROC-AUC** co-primary tiebreak (decision D8). When CV-F1 ties
  within its fold std, break on ROC-AUC — and record that you did so *before* looking at the test.
- **Metrics from probabilities:** AUROC/AUPRC are computed from predicted **probabilities**, threshold
  metrics (Acc/Prec/Rec/F1) from the 0.5-thresholded probability. The model must output probabilities
  (not hard labels), or AUROC/AUPRC degenerate.
- **Feature-set ablation:** you may report every feature set's 278-test metric, but the **headline model
  is the CV-selected one** — never re-pick the headline by the test score (that is test-set selection).
- **Seeds:** `random_state = 1`, `K_FOLDS = 5`, `VAL_FRACTION = 0.1`. For a *matched* CV comparison,
  build folds with the identical `StratifiedKFold(n_splits=5, shuffle=True, random_state=1)` on the same
  train parquet (same row order) → DGT sees the same folds as the MPNN.

**Anti-patterns explicitly removed here (avoid them in DGT):**
1. **Test-driven grid search.** The old `grid_search_train_test` (loop the grid, keep the combo with the
   best *test* F1) was dropped — it leaks the test into selection. Kept as a learning-record note in
   `sk_runner.py`.
2. **Warm-start "CV".** chemprop v2's `train_model_cv` (in `m_chemprop/.../training.py`) is *repeated
   percentage splits with each fold loading the previous fold's weights* — not a leak-free k-fold. The
   `kfold_cv/` module replaces it (fresh model per fold, proper StratifiedKFold). If DGT-CDI has an
   analogous "cv = repeated split + warm-start" routine, **do not use it for model selection.**
3. **Silent hard-label metrics.** e.g. sklearn `SVC(probability=False)` has no `predict_proba`, so an
   AUROC computed from `predict()` is degenerate. Ensure DGT emits calibrated probabilities.

**Non-determinism (carry this lesson).** GPU training here is non-deterministic (no fixed seed): the
*same* MPNN config scored CV-mean F1 78.20 vs 79.09 across two runs — a run-to-run σ comparable to the
entire HPO grid spread. So (a) report CV as **mean ± std over folds**, (b) don't over-interpret small
HPO ranking gaps (they can be pure noise — our tuning "winner" was just the default), and (c) for
reproducible finals, set a seed (`pl.seed_everything`) and/or average a few seeds. Expect the same for
DGT (transformer on GPU).

---

## 3. What has been done here (snapshot for comparison)

All numbers are on the **same 278-test**; selection was always by train CV.

- **sklearn baselines** (`run_baselines.py`, RF/MLP/SVM/HGB × qm/rdkit_fg/qm_rdkit, 5-fold GridSearchCV):
  best = **HGB × rdkit_fg** (F1 85.00, AUROC 91.52, AUPRC 92.25); best AUROC/AUPRC = HGB × qm_rdkit
  (91.85 / 92.49). Highest CV-F1 cell = HGB × rdkit_fg (0.813), which is also the test-best → leak-free.
- **MPNN (chemprop 2.2.3)** proper 5-fold CV (`kfold_cv/`): CV-F1 a tie across feature sets (77.7–78.5),
  broken on AUROC → headline **qm_rdkit**. 278-test: qm_rdkit F1 83.62 / AUROC 89.11; rdkit_fg F1 85.22 /
  AUROC 89.69; qm F1 84.62 / AUROC 89.69.
- **Cross-model (matched CV):** MPNN CV-F1 ≈78.4 < HGB CV-F1 ≈81.3 → descriptor gradient-boosting leads
  the MPNN on this endpoint (endpoint is composition/functional-group + size driven; a graph view
  under-represents it). **DGT's job is to test whether a transformer graph model closes that gap.**
- **Artifacts:** HGB models at `s3://cdi-lab-workspaces/ts_project_1/models/biodegradation/GWU/`
  (`model_hgb_biodeg_gwu_no_ind_<fs>.joblib` + paired `_scaler.joblib`); MPNN `.pt` in
  `models_trained/gwu_paper/`. Full tables in `documents/project/param_optimization.md`.

---

## 4. Relevant repo layout (trans_learn)

```
src/trans_learn/
├── settings.py                         # DATASET_REGISTRY (paths/config), PATH_ROOT
├── utils/
│   ├── load_data.py                    # load_split_data(split, dataset_name) -> {ids_ys, descs}
│   └── sk_train_tools.py               # grid_search_sk / train_sk / pred_sk (proba path; saves StandardScaler)
├── models/
│   ├── other_models/classification/    # ---- sklearn ----
│   │   ├── sk_runner.py                # shared runner: data load + name-based feature select + CV + final
│   │   ├── randforest.py, mlp.py, svm.py, hgb.py   # thin configs (RunConfig: factory+default+grid)
│   │   ├── run_baselines.py            # resumable model×feature_set matrix -> baseline_results.{json,md}
│   │   └── lgbm_legacy.py, gbrt_legacy.py, gbc.py  # legacy / excluded (SHAP routines live in lgbm_legacy)
│   └── m_chemprop/classification/      # ---- MPNN ----
│       ├── training.py, testing.py     # existing single-config trainer (NOT leak-free CV; left as-is)
│       ├── analysis_plot.py            # evaluate_binary_classification (the metric definitions)
│       ├── pipeline_config.py          # config for the existing training.py workflow
│       └── kfold_cv/                   # <-- leak-free module = the template to mirror for DGT
│           ├── cv_config.py            # dataset + arch (DEFAULT_MPNN_PARAMS) + HPO_GRID + output paths
│           ├── mpnn_common.py          # shared helpers (data->datapoints, loaders+scaler, build+fit, CV metrics)
│           ├── run_cv.py               # 5-fold CV over feature sets (resumable) -> mpnn_cv_results.{json,md}
│           ├── run_final.py            # train-on-all + score 278-test ONCE -> mpnn_final_results.{json,md}
│           └── run_hpo.py              # architecture screening (resumable) -> mpnn_hpo_results.{json,md}
tests/utils/test_utils.py               # data-load + feature-selection sanity tests
models_trained/gwu_paper/               # saved models (.joblib / .pt)
data/pred_data/gwu_paper/               # predictions + *_results.{json,md} reports
```

---

## 5. Code patterns to port (mirror `kfold_cv/` for DGT)

The `kfold_cv/` module is the clean, leak-free template. Recreate the same 5-file shape in `DGT-CDI`
(swap "mpnn" for "dgt"), keeping the orchestration identical:

| trans_learn file | role | DGT equivalent to create |
|---|---|---|
| `cv_config.py` | dataset name, split/seed constants, `DEFAULT_*_PARAMS`, `HPO_GRID`, output paths | `dgt_cv_config.py` |
| `mpnn_common.py` | load split → datapoints/graphs, build model, train fold, predict **probabilities**, compute metrics | `dgt_common.py` |
| `run_cv.py` | StratifiedKFold(5) on train, fresh model/fold, CV-mean F1/AUROC, resumable JSON+MD | `run_cv.py` |
| `run_final.py` | train on ALL train, **score 278-test once**, save model | `run_final.py` |
| `run_hpo.py` | expand `HPO_GRID`, run the CV per config, ranked report, resumable | `run_hpo.py` |

Key reusable patterns:
- **Resumable orchestration:** write a `results.json` after *each* cell/config; re-run skips finished
  ones (Ctrl-C / power-off safe). Render a `.md` report alongside.
- **One params dict** (`DEFAULT_*_PARAMS`) threaded through build/fit/CV/final/HPO, so `run_final`
  automatically uses the HPO winner once it's locked into the config.
- **Identical metrics:** reuse `evaluate_binary_classification` (Acc/Prec/Rec/F1 at 0.5; AUROC/AUPRC from
  probabilities) — or replicate its exact definitions in DGT — so numbers are directly comparable. It
  treats the `pred` column as a **probability**.
- **Descriptor scaler fit on train only:** if DGT ingests `qm`/`rdkit_fg` descriptors, fit the
  StandardScaler/normalizer on the training fold and apply to val/test (never fit on test). If DGT is
  graph-only, it corresponds to the `none` feature set.

---

## 6. DGT retrofit checklist

- [ ] Load `biodeg_gwu_b2_no_ind_{train,test}.parquet` (same fixed split; **no re-split**). Target
      `degradable`, SMILES `smiles`.
- [ ] Decide DGT's feature sets: at least `none` (graph only); add `qm`/`rdkit_fg`/`qm_rdkit` iff DGT
      can ingest molecular descriptors (same suffix selection as §1).
- [ ] Implement `run_cv.py`: `StratifiedKFold(5, shuffle=True, random_state=1)` on TRAIN; fresh DGT per
      fold; inner val from train folds; fixed epochs (or early-stop on the inner val); CV-mean F1/AUROC.
- [ ] Select the feature set / arch by **CV-mean F1** (AUROC tiebreak), committed **before** any test use.
- [ ] Implement `run_final.py`: retrain on all train with the selected config, **score the 278-test
      once**, save the model + emit metrics.
- [ ] Implement `run_hpo.py` for architecture/hyperparameter screening on the selected feature set only
      (leak-free CV; resumable; ranked report).
- [ ] Ensure DGT outputs **probabilities**; compute metrics with the shared definitions.
- [ ] Record results in a `param_optimization.md`-style file and upload the final model to
      `s3://cdi-lab-workspaces/ts_project_1/models/biodegradation/GWU/` (+ any scaler).
- [ ] Compare to HGB / MPNN on the same 278-test and the same CV basis.

---

## 7. Test-data leakage audit (verify in DGT-CDI)

Grep the DGT code and confirm each:

1. **Selection never sees test.** Architecture/HPO/feature-set/epoch/threshold choices are made only from
   train-CV scores. Search for any place the 278-test (or `test`/`val==test`) feeds a selection decision.
2. **Test scored once.** The test loader is built and predicted exactly once, in the final script — not
   inside the CV/HPO loops.
3. **No fit-on-test.** Scalers/normalizers, feature selection, PCA, target stats, class weights, and the
   decision threshold are all fit on **train only** (train folds during CV), then applied to val/test.
4. **Fresh model per fold.** No weights/optimizer state carried across folds (watch for any warm-start
   "cv = repeated split + load previous checkpoint" pattern — that leaks and biases fold scores).
5. **Inner val ⊂ train.** Early-stopping / monitoring uses a validation split carved from the training
   folds, never the held-out test.
6. **Respect the external split.** If DGT does its own internal train/val/test split, make sure it does
   **not** re-partition the full pool (which would mix the fixed test back in). Use the provided
   train/test parquet as the boundary.
7. **Probabilities, not hard labels**, into AUROC/AUPRC (else degenerate and non-comparable).
8. **No de-dup re-split needed:** the split-leakage audit is already resolved (0 true cross-split
   duplicates; fingerprint-identical pairs are homologs/stereoisomers). Keep the fixed split; do not
   "clean" the test against train.

---

## 8. Cross-model comparison rules (so DGT vs MPNN vs HGB is fair)

- Same 278-test, same metric definitions, same 0.5 threshold, probabilities for AUROC/AUPRC.
- Selection by train CV for every model family (report CV-mean too, for the apples-to-apples read).
- Report a feature-set ablation but keep a single CV-selected headline per family.
- State seeds, folds, epochs, and package versions for reproducibility (paper §15).

Reference numbers to beat / match are in §3 and in `documents/project/param_optimization.md`.
