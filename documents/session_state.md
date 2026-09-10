# Session state

> Living "you-are-here" doc. Updated at the end of each session before context auto-compacts.
> Durable docs ([overview.md](overview.md), [tech.md](tech.md), [modeling_routine.md](modeling_routine.md), [trained_models.md](trained_models.md), [config_reference.md](config_reference.md), [graph_transformer.md](graph_transformer.md), [dgt_porting_guide.md](dgt_porting_guide.md), [upstream_sync.md](upstream_sync.md)) describe *how* the project works. This doc captures *where it is right now*.
> PR-in-progress records: [adr/0001-pr-mol-desc.md](adr/0001-pr-mol-desc.md), [log/pr-1-mol-desc.md](log/pr-1-mol-desc.md), [projects/gwu.md](projects/gwu.md), [projects/paper.md](projects/paper.md).

**Last updated:** 2026-09-09

---

## Current focus — `biodeg_gwu_no_ind` (2026-08-31 → 09-09)

**Status: the science is done and both models are deployed.** What remains is two decisions
(§8 framing, and merging the branch) plus optional polish. Full write-up:
**[projects/paper.md](projects/paper.md)** — methods, results, per-seed and per-fold values,
leakage audit, limitations, future work.

### Result

Canonical dataset: 5264 train / 526 val / 278 test, InD removed, 247 descriptors
(40 QM `_gwu` + 207 RDKit/fg).

- **Selected config:** `BiodegNoInd-DGT-Pipeline-WithDesc-nongwu` (`rdkit_fg`, 207 desc).
  Recorded 2026-09-02 on validation with test suppressed; **confirmed independently by
  stratified 5-fold CV** (all four arms tie on F1 → AUC tiebreak → same config).
- **Test, read once:** F1 0.8610 ± 0.0066, ROC-AUC 0.9196 ± 0.0027, AUPRC 0.9269 ± 0.0051
  (4 seeds, threshold 0.5).
- **Three publishable findings** (paper.md §7):
  1. **QM descriptors are significantly weaker than RDKit/fg** and add nothing on top of them
     — ΔF1 −0.0091, **0/5 folds**, t = −7.36, p ≈ 0.002. The only significant effect found.
  2. **F1-argmax thresholds are unidentifiable** at n = 278 — plateau spans ~half the
     probability scale; the argmax sits on one molecule's score, and nudging it 1.5e-5 moves
     F1 by 0.004. Thresholds across checkpoints spanned 0.357–0.667 with ROC-AUC unchanged.
  3. **A single seed re-run reordered the four-arm ranking** and changed which config the
     protocol selects (§6.1) — 0.0127 F1 on one seed, larger than the whole between-arm spread.
- **Descriptor benefit: not established.** Paired across folds, `rdkit_fg` − `none` is
  ROC-AUC +0.0035 (p ≈ 0.09, 4/5 folds) and F1 +0.0009 (n.s.). The +0.0183 reported in
  [projects/gwu.md](projects/gwu.md) does not replicate — though that comparison also spans a
  dataset change, which is why it is *not* the strongest evidence (see framing doc).

### Deployed — all four bundles in S3, verified 9 objects each (2026-09-03)

`s3://cdi-lab-workspaces/ts_project_1/models/biodegradation/GWU/<bundle>/`
and locally in `results/final_models/`. Index with full provenance:
**[../results/final_models/INDEX.md](../results/final_models/INDEX.md)**.

| Bundle | Feature set | Budget | Threshold |
|---|---|---|---|
| `biodeg-no-ind-dgt-nongwu-2026-09-03` | `rdkit_fg` (207) | 29 ep (CV) | 0.5 |
| `biodeg-no-ind-dgt-graphonly-2026-09-03` | `none` | 33 ep (CV) | 0.5 |

Both `2026-09-02` bundles are superseded (seed-derived budgets) but retained. Three undated
legacy bundles are pre-protocol (test-selected seeds, argmax thresholds) — flagged in INDEX.md
as do-not-cite.

**Deploy `graphonly`** unless the descriptor variant is specifically needed: it takes SMILES
only, while `nongwu` requires the caller to supply 207 RDKit/fg columns in training order. The
two are statistically indistinguishable.

### Built across this session

- **Dataset:** `biodeg_gwu_no_ind` loader, format registration, 4 configs.
- **CV harness:** `split_mode: cv-train-<k>` (folds train+val only, test untouched,
  `random_state=1` per guide §2) + `scripts/cv/{dgt_cv_config,dgt_common,run_cv}.py` —
  resumable sweep, §2 selection rule, JSON+MD report. Artifacts in `results/DGT_cv/`.
- **Leak-free selection tooling:** `scripts/rank_configs_by_val.py` (val-based ranking,
  dataset guard, `--metric` override, `--hide-test`); median-**val** seed choice in
  `retrain_on_trainval.py` (+ `--epochs` override); `dgt_train.py` dumps `val/predictions.pt`;
  `analyze_run.py` fits the threshold on val; `retrain_on_trainval.py` writes
  `best_f1_threshold: null` rather than inheriting a training seed's value, and `predict.py`
  fails loudly on null.
- **Docs:** [projects/paper.md](projects/paper.md),
  [projects/paper_framing_options.md](projects/paper_framing_options.md),
  [upstream_sync.md](upstream_sync.md), [../results/final_models/INDEX.md](../results/final_models/INDEX.md),
  modeling_routine Step 0 + rewritten Step 6, project gotchas in [../CLAUDE.md](../CLAUDE.md),
  supersession banners on [trained_models.md](trained_models.md) and
  [projects/gwu.md](projects/gwu.md).
- **`.gitignore`:** `results/` now tracks exactly one file, `final_models/INDEX.md` (tested
  against a 33-file dummy tree).

### Next actions

1. **Merge `mol-desc` → `main`.** 24 commits plus 6 uncommitted files
   (`.gitignore`, `paper.md`, `trained_models.md`, `paper_framing_options.md`,
   `results/final_models/INDEX.md`). `main` has none of this work.
   [upstream_sync.md](upstream_sync.md) §5: leaving it stranded means resolving the same
   `master_loader.py` / `san_graph.py` conflicts twice if upstream ever sends an update.
2. **Decide the §8 framing** — the only open item that changes the paper's shape. Both drafts
   with costs in [projects/paper_framing_options.md](projects/paper_framing_options.md);
   recommendation is Option C now, B later.
3. **Unverified housekeeping** (may already be done): back up `results/` metadata
   (`tar --exclude='*.ckpt'`) to S3, since `results/DGT_cv/dgt_cv_results.{json,md}` is cited
   in paper.md §10 and exists on one machine only; and confirm the `biodeg_gwu_no_ind` entry in
   trans_learn's `settings.py` is committed in that repo.
4. Everything else is optional — paper.md §12 Future work.

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

## 🚩 Next-session task order — SUPERSEDED (2026-06-10)

> ⚠️ **Historical.** This block and the one below plan the `biodeg_gwu` work and were written
> before the `biodeg_gwu_no_ind` study. Items 1–2 were completed on a different dataset;
> item 5 (merge `mol-desc`) is still open and now carries 24 commits.
> **For current next actions see [Next actions](#next-actions) at the top of this file.**

<details>
<summary>Expand the 2026-06-10 plan</summary>


1. **Confirm the non-GWU deployment bundle.** Run (if not already): `rm datasets/biodeg_gwu/processed/data_stdesc_fa7b7fe0.pt` → `python scripts/retrain_on_trainval.py results/DGT/Biodeg-GWU-DGT-Pipeline-WithDesc-nongwu/` → verify `final_model.json` has `descriptor_columns` (207) + `desc_stats`. Optional end-to-end predict on the S3 test parquet (command in [projects/gwu.md](projects/gwu.md) / chat).
2. **Record the non-GWU winner** in [trained_models.md](trained_models.md) (Final models table).
3. **Group F docs:** [tech.md](tech.md) (add `line_graph_with_desc` head to Stage-4 readout), [config_reference.md](config_reference.md) (`dataset.desc_dim` / `standardize_desc` / `desc_include`/`exclude`/`columns`, `gnn.desc_proj_dim`).
4. **(optional)** descriptors-only MLP sanity baseline (Phase-4 leftover).
5. **Merge decision** for branch `mol-desc` → main (PR-1), once F + trained_models.md row are done.
6. **(optional)** repeat the descriptor-type study on biodeg (no-Reaxys) if wanted.

## Where to start next session

Pick up at **Next-session task order #1** (confirm the non-GWU bundle / cache rebuild) — that's the only thing with a loose end. Everything else (docs, MLP baseline, merge) is independent and can be done in any order. All Phase-2 code is implemented, compiles, and has been run on the remote; no code is mid-edit.

</details>
