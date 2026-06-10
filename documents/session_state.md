# Session state

> Living "you-are-here" doc. Updated at the end of each session before context auto-compacts.
> Durable docs ([overview.md](overview.md), [tech.md](tech.md), [modeling_routine.md](modeling_routine.md), [trained_models.md](trained_models.md), [config_reference.md](config_reference.md), [graph_transformer.md](graph_transformer.md)) describe *how* the project works. This doc captures *where it is right now*.
> PR-in-progress records: [adr/0001-pr-mol-desc.md](adr/0001-pr-mol-desc.md), [log/pr-1-mol-desc.md](log/pr-1-mol-desc.md), [projects/gwu.md](projects/gwu.md).

**Last updated:** 2026-06-10 (end of session)

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
