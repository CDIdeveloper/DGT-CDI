# Session state

> Living "you-are-here" doc. Updated at the end of each session before context auto-compacts.
> Durable docs ([overview.md](overview.md), [tech.md](tech.md), [modeling_routine.md](modeling_routine.md), [trained_models.md](trained_models.md), [config_reference.md](config_reference.md), [graph_transformer.md](graph_transformer.md), [dgt_porting_guide.md](dgt_porting_guide.md), [upstream_sync.md](upstream_sync.md)) describe *how* the project works. This doc captures *where it is right now*.
> PR-in-progress records: [adr/0001-pr-mol-desc.md](adr/0001-pr-mol-desc.md), [log/pr-1-mol-desc.md](log/pr-1-mol-desc.md), [projects/gwu.md](projects/gwu.md), [projects/paper.md](projects/paper.md).

**Last updated:** 2026-09-02

---

## Current focus — `biodeg_gwu_no_ind` (2026-08-31 → 09-02)

**Where we are:** the canonical dataset (5264 train / 278 test, InD removed) is onboarded and
the 4-arm descriptor ablation is complete. The winner was selected **on validation with test
suppressed** — the first selection in this project made under the porting-guide §2 protocol.
Full write-up: **[projects/paper.md](projects/paper.md)** (paper base doc: methods, results,
per-seed values, leakage audit, limitations).

- **Selected config:** `BiodegNoInd-DGT-Pipeline-WithDesc-nongwu` (207 RDKit/fg descriptors).
  F1 top-two tied within seed std (0.8165 vs 0.8164) → broken on ROC-AUC (0.8876 vs 0.8853).
  Recorded 2026-09-02, before any test number was read.
- **Headline finding:** the descriptor channel is ~neutral here (best F1 +0.0050 over
  graph-only, ROC-AUC +0.0001) — the earlier +0.0183 in [projects/gwu.md](projects/gwu.md)
  does **not** replicate once selection moves off the test set.
- **Confirmed by 5-fold CV (2026-09-02).** The §2 protocol is now implemented
  (`split_mode: cv-train-<k>` + `scripts/cv/`) and run: all four arms tie on F1, AUC tiebreak
  selects `rdkit_fg` — the same config the single-split selection recorded. Artifacts in
  `results/DGT_cv/`. Paired fold-by-fold, descriptors give F1 +0.0009 (3/5 folds) and
  ROC-AUC +0.0035 (4/5 folds, p ≈ 0.09) over graph-only: **no established benefit.**
- **Test read once** for the selected config: F1 0.8610 ± 0.0066, ROC-AUC 0.9196 ± 0.0027,
  AUPRC 0.9269 ± 0.0051.
- **Two deployment bundles built** (`rdkit_fg` and graph-only) — see
  [projects/paper.md](projects/paper.md) §10.1. Graph-only ships alongside because it needs
  only SMILES at inference; on this dataset the two are indistinguishable.

**Built this session:** `biodeg_gwu_no_ind` loader + format registration + 4 configs;
[scripts/rank_configs_by_val.py](../scripts/rank_configs_by_val.py) (validation-based config
ranking, dataset guard, F1/AUC override); [upstream_sync.md](upstream_sync.md) (fork
provenance + merge checklist); modeling_routine Step 0 (dataset onboarding) and a rewritten
Step 6 (select on validation, not test); dataset/test-selection warnings on
[trained_models.md](trained_models.md) and [projects/gwu.md](projects/gwu.md).

**Built this session (continued):** `split_mode: cv-train-<k>` (folds train+val only, test
untouched) and `scripts/cv/` (`dgt_cv_config.py`, `dgt_common.py`, `run_cv.py` — resumable
5-fold sweep, §2 selection rule, JSON+MD report); median-**val** seed choice in
`retrain_on_trainval.py`; validation-fitted decision thresholds
(`dgt_train.py` dumps `val/predictions.pt`, `analyze_run.py` consumes it); project gotchas in
[../CLAUDE.md](../CLAUDE.md).

**Next actions, in order:**
1. Copy both bundles to `results/final_models/` and upload to S3 (paper.md §10.1 has the
   URIs). Verify each manifest's `best_f1_threshold` is the value measured on that deployed
   checkpoint, not one inherited from a training seed.
2. Optional: rebuild the `rdkit_fg` bundle at the CV-derived 29-epoch budget
   (`retrain_on_trainval.py --epochs 29`) instead of the 22 taken from one seed's val curve.
3. Open items in paper.md §11 — AUPRC done, CV done; what remains is the descriptor-effect
   question (ROC-AUC +0.0035, p ≈ 0.09), whether to add seeds within folds, and the §8
   framing decision.

**Note:** everything below this section predates the `biodeg_gwu_no_ind` work and refers to
the older `biodeg_gwu` dataset (300-row test, test-selected). Kept for history.

---

## Where we are

- **Phase 0 / Phase 1** — closed. biodeg_gwu baseline AUC 0.8821 ± 0.0034 (HPO round-1 winner); biodeg (no-Reaxys) baseline AUC 0.9007 ± 0.0024. Both in [trained_models.md](trained_models.md).
- **Phase 2 — descriptor late-fusion** — **implemented & verified on branch `mol-desc` (PR-1, NOT merged to main).** Standardisation, `line_graph_with_desc` head, descriptor-column selection, and `predict.py` descriptor support (group G) all done and run on the remote.
- **gwu descriptor-type study** — **COMPLETE** (5 variants + a `desc_proj_dim` sweep). Full results/conclusions in [projects/gwu.md](projects/gwu.md).
- **Remaining for PR-1:** group F **docs** (tech.md head + config_reference fields); descriptors-only MLP baseline; trained_models.md final-model row; merge decision.

## Key results — biodeg_gwu descriptor-type study (controlled: same backbone, only descriptor channel changes)

| Variant | desc_dim | Test AUC | Δ vs baseline |
|---|---|---|---|
| none (baseline) | — | 0.8821 ± 0.0034 | — |
| all | 247 | 0.8966 ± 0.0027 | +0.0145 |
| GWU/QM only | 40 | 0.8728 ± 0.0070 | −0.0093 |
| **non-GWU (RDKit)** | 207 | **0.9004 ± 0.0004** | **+0.0183 (best)** |
| SHAP-selected (90% cov) | 94 | 0.8864 ± 0.0055 | +0.0043 |

`desc_proj_dim` sweep on GWU-only (16/32/64/128) → all ~0.873–0.876, none reach baseline (signal problem, not capacity). **Conclusion: the RDKit/non-GWU descriptors carry the gain; GWU/QM descriptors don't help.** Recommended deployable = **non-GWU**.

## What was delivered this session (2026-06-09 → 06-10)

- **Pipeline (Phase 2):** `dataset.desc_dim` / `dataset.standardize_desc` / `gnn.desc_proj_dim` config fields; `LineGraphWithDescHead` (`@register_head('line_graph_with_desc')`, post-readout `Linear(desc_dim→desc_proj_dim)→GELU` fused before `out_layer`); loader train-split z-score → **separate processed cache** + `desc_stats{suffix}.json`; `tests/test_descriptor_head.py`.
- **Descriptor selection:** `dataset.desc_include`/`desc_exclude`/`desc_columns` (precedence columns>include>all, exclude last); auto-hash cache keying ([_desc_select.py](../graphgps/loader/dataset/_desc_select.py)); `tests/test_desc_select.py`; [scripts/select_features_from_shap.py](../scripts/select_features_from_shap.py) (SHAP CSV from S3 → `desc_columns`); configs `*-WithDesc{,-gwu,-nongwu,-gwu-descdim-16/32/64}.yaml` + `Biodeg-DGT-Pipeline-WithDesc.yaml`.
- **Group G (predict):** `predict.py` handles descriptors (reads `descriptor_columns`+`desc_stats` from `final_model.json`, validates+reorders input cols by name, applies z-score); `retrain_on_trainval.py` embeds `descriptor_columns`+`desc_stats` into `final_model.json`; **desc_stats filename made suffix-keyed (collision fix)**. `predict.py` input now accepts **local or `s3://`, CSV or Parquet**.
- **Refactor/tools:** `_mol_featurise.py` (shared featurisation); `analyze_run.py --no-plots`; `predict.py --label-col` eval metrics+plots via shared `_eval_plots.py`.
- **Docs:** ADR 0001; pr-1 log; projects/gwu.md; overview.md Phase 2/3/4 marked done; tech.md data-lineage; modeling_routine Quickstart.

## ⚠️ Known gotchas / must-read before next actions

1. **Stale `desc_stats` caches.** Selection caches built *before* the suffix-keying fix have their stats under the old fixed name `desc_stats.json` (overwritten across selections). **To deploy any descriptor variant:** `rm datasets/biodeg_gwu/processed/data_stdesc_<hash>.pt`, then re-run that variant's `retrain_on_trainval.py` — the rebuild writes `desc_stats_<hash>.json`, which retrain then embeds into `final_model.json`. **non-GWU hash = `fa7b7fe0`.** (The rebuild is byte-identical data; reported metrics stay valid.)
2. **predict.py S3 input** = full `s3://cdi-lab-workspaces/<key>` URI (needs s3fs + AWS creds — present). A bare key is treated as a local path.
3. **Descriptor models at predict time** need the descriptor columns in the input table; extra columns (e.g. the 40 `_gwu` in the full test parquet) are fine — selected by name, rest ignored, all preserved in output.

## 🚩 Next-session task order

1. **Confirm the non-GWU deployment bundle.** Run (if not already): `rm datasets/biodeg_gwu/processed/data_stdesc_fa7b7fe0.pt` → `python scripts/retrain_on_trainval.py results/DGT/Biodeg-GWU-DGT-Pipeline-WithDesc-nongwu/` → verify `final_model.json` has `descriptor_columns` (207) + `desc_stats`. Optional end-to-end predict on the S3 test parquet (command in [projects/gwu.md](projects/gwu.md) / chat).
2. **Record the non-GWU winner** in [trained_models.md](trained_models.md) (Final models table).
3. **Group F docs:** [tech.md](tech.md) (add `line_graph_with_desc` head to Stage-4 readout), [config_reference.md](config_reference.md) (`dataset.desc_dim` / `standardize_desc` / `desc_include`/`exclude`/`columns`, `gnn.desc_proj_dim`).
4. **(optional)** descriptors-only MLP sanity baseline (Phase-4 leftover).
5. **Merge decision** for branch `mol-desc` → main (PR-1), once F + trained_models.md row are done.
6. **(optional)** repeat the descriptor-type study on biodeg (no-Reaxys) if wanted.

## Where to start next session

Pick up at **Next-session task order #1** (confirm the non-GWU bundle / cache rebuild) — that's the only thing with a loose end. Everything else (docs, MLP baseline, merge) is independent and can be done in any order. All Phase-2 code is implemented, compiles, and has been run on the remote; no code is mid-edit.
