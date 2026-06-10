# PR #1: Molecular descriptor late-fusion (Phase 2)

<!-- DRAFT — for iteration -->

## Meta
- PR: #1
- Title: Molecular descriptor plumbing (late fusion at the head)
- Head branch: mol-desc
- Base branch: main
- Status: **Complete — ready to merge.** Phase-2 pipeline (A–E) + gwu study (5/5) + group G (predict.py descriptors) + **group F (docs)** all done; trained_models.md final-model row added. Descriptors-only MLP baseline moved to overview.md → Future work. Follow-up: `desc_proj_dim`/`dim_hidden` exploration (#6).
- Related ADRs:
  - [../adr/0001-pr-mol-desc.md](../adr/0001-pr-mol-desc.md)

## Summary
One paragraph: add a `line_graph_with_desc` readout head that concatenates the standardised molecular descriptor vector (`batch.desc`, `[B, desc_dim]`) with the pooled atom/bond embeddings before the final MLP, leaving the DGT backbone untouched. Enables the Phase-4 ablation (graph-only vs graph+descriptors) via a single `gnn.head` config toggle. Descriptors are z-scored using train-split stats, persisted for leak-free val/test/inference.

## TODO (PR checklist)

> **Status (2026-06-10):** Groups **A, B, C, D, E, F, G = DONE** (+ descriptor-selection extension + gwu study + S3/parquet predict input). trained_models.md final-model row added. Descriptors-only MLP baseline moved to overview.md → Future work. **All checklist items below are ticked; PR is ready to merge.**

**Implementation order:** A (config) → B (loader standardisation) → C (head) → D (minimal test) → E (WithDesc config + 3-epoch dry-run) → G (predict.py descriptors) → F (docs). The group-E dry-run is the primary integration check.

### A. Config plumbing (do first — GraphGym rejects unknown YAML keys)
- [x] Register `cfg.dataset.desc_dim` (int, default `0`) in [graphgps/config/dataset_config.py](../../graphgps/config/dataset_config.py).
- [x] Register `cfg.dataset.standardize_desc` (bool, default `False`) in the same file — when True, the loader z-scores desc and writes a **distinct** processed cache (separate from the baseline's raw cache).
- [x] Register `cfg.gnn.desc_proj_dim` (int, default `128`) — output dim of the head's descriptor MLP `f(desc)`; the **tunable** knob to modulate descriptor influence (model-only → no cache invalidation when swept).
- [x] Sanity: a config setting `dataset.desc_dim: 216` + `dataset.standardize_desc: True` + `gnn.desc_proj_dim: 128` loads without a yacs error.

### B. Descriptor standardisation — RESOLVED 2026-06-09 (Option A; see ADR)
- [x] In `process()` ([biodeg.py](../../graphgps/loader/dataset/biodeg.py) + [biodeg_gwu.py](../../graphgps/loader/dataset/biodeg_gwu.py)): compute z-score μ/σ from the **train split only**; apply to all splits; guard zero-variance cols (`std + eps`).
- [x] Persist μ/σ **and descriptor column NAMES** to a per-dataset `desc_stats.json`.
- [x] Write the normalised result to a **separate processed cache** — do NOT overwrite / `rm` the baseline's raw-desc cache (new model → new cache). Mechanism TBD in group A/E (config-driven; see open item).
- [x] At retrain, propagate `descriptor_columns` + `desc_stats` into `final_model.json` (predict.py becomes self-contained). (Inference handling → group G.)

### C. DescriptorGraphHead
- [x] Add `DescriptorGraphHead` under [graphgps/head/](../../graphgps/head/), mirroring `LineGraphHead` ([san_graph.py:57](../../graphgps/head/san_graph.py#L57)). Keep the per-node FC layers + atom/bond pooling **identical**; the descriptor enters **after the readout (pooling)** — concat `f(desc)` with the pooled `[atom ‖ bond]` vector just before `out_layer` (matches the DGT author's "add after the readout").
- [x] Descriptor MLP `f(desc)` = `Linear(cfg.dataset.desc_dim → cfg.gnn.desc_proj_dim) → GELU`. `out_layer` input dim becomes `2·(dim_in // 2**L) + desc_proj_dim`.
- [x] Register `@register_head('line_graph_with_desc')`.
- [x] Add `MOLECULAR DESCRIPTORS CONSUMED HERE` marker at the `batch.desc` read site.
- [x] Verify marker invariant: `grep -rn 'MOLECULAR DESCRIPTORS' graphgps/ scripts/` → exactly 2 ENTER (biodeg + biodeg_gwu) + 1 CONSUMED.

### D. Tests (kept minimal — primary verification is the group-E dry-run)
- [x] **One cheap unit test** `tests/test_descriptor_head.py`: instantiate `line_graph_with_desc`, forward a toy batch, assert output `[B, C]`, assert both `line_graph` + `line_graph_with_desc` register. Runs in seconds, no GPU; catches head shape bugs that are painful to diagnose from a full run.
- [x] After the loader change, re-run existing `pytest -rP tests/test_dataset.py` **once** (confirms standardisation didn't break the loaders / desc stays finite).
- [x] Everything else: rely on the end-to-end 3-epoch dry-run (group E). No per-step tests; broader checking deferred to the full 4-seed run.

### E. Config + end-to-end smoke (bridges into Phase 3)
- [x] `configs/biodegradability/Biodeg-DGT-Pipeline-WithDesc.yaml` — `gnn.head: line_graph_with_desc`, `dataset.desc_dim: 216`, `dataset.standardize_desc: True`. (Baseline `Biodeg-DGT-Pipeline.yaml` stays untouched.)
- [x] (Optional) `Biodeg-GWU-DGT-Pipeline-WithDesc.yaml` — `dataset.desc_dim: 247`, `dataset.standardize_desc: True`.
- [x] 3-epoch dry-run confirms the pipeline runs with the desc head (exit 0, val AUC > 0.5).

### G. Inference with descriptors (predict.py, WithDesc model only)
- [x] Input CSV must carry the descriptor columns — descriptors are **not** derivable from SMILES (esp. biodeg_gwu QM features); predict.py reads them from the CSV, not from RDKit.
- [x] Validate input descriptor columns against `final_model.json` → `descriptor_columns`: error listing any **missing** required columns; warn on extras.
- [x] **Reorder** input descriptor columns to the exact **training order** before building `Data.desc` (column order matters to the model; name-based reorder makes the user's CSV column order irrelevant as long as all named columns are present).
- [x] Apply persisted `desc_stats` (μ/σ) from `final_model.json` — identical normalisation to training.
- [x] Branch on the bundle's config: only the `gnn.head: line_graph_with_desc` path consumes descriptors; the baseline SMILES-only path is unchanged.

### F. Docs
- [x] Tick [overview.md → Phase 2](../overview.md#phase-2--descriptor-plumbing-late-fusion) checkboxes as completed.
- [x] [tech.md](../tech.md) — add the desc flow + `line_graph_with_desc` readout variant (Stage 4).
- [x] [config_reference.md](../config_reference.md) — document `dataset.desc_dim` + the new head value.
- [x] [session_state.md](../session_state.md) — update at end of session.

## Context
- Problem / motivation: incorporate molecular descriptors into DGT to test whether they add signal on top of graph features (Phase 2 → Phase 4 ablation).
- Constraints: backbone untouched for a clean ablation; no train→val/test leakage in normalisation; minimal surgical changes.
- Non-goals: early/mid fusion (deferred escalation); solvent fusion (other datasets); HPO of the descriptor-augmented model (later).

## Approach
- High-level plan: config field → loader standardisation → new head → tests → WithDesc config → dry-run → ablation (Phase 4).
- Key design notes: see [ADR 0001](../adr/0001-pr-mol-desc.md). Late fusion at the head; `desc` carried untouched through the backbone; toggle via `gnn.head`.

### Data processing flow (where descriptors live at each stage)
```
trans_learn DATASET_REGISTRY (mirror: tests/data_loading/settings.py)   # per-dataset: S3 paths, id_column_count, target_column
   ─DatasetLoader (tests/data_loading/load_data.py)─▶ read S3 parquet, clean, split cols  # first id_column_count = SMILES/ids/y ; REST = descriptors (biodeg=5, gwu=10)
   ─prepare_data.py─▶ datasets/<name>/raw/{train,test}.parquet (+ manifest.json: descriptor_columns, desc_dim)   # raw desc, unnormalised
   ─<loader>.process()─▶ datasets/<name>/processed/data*.pt   (Data.desc [1, desc_dim])                          # train.parquet→90/10 train/val ; test.parquet→test
        └ z-score desc, TRAIN-split μ/σ ─▶ desc_stats.json     # Phase 2 (Option A): normalise + persist μ/σ + column NAMES; SEPARATE processed cache
   ─main.py (train.mode: dgt)─▶ results/DGT/<cfg>/<seed>/{ckpt, test/predictions.pt}
   ─analyze_run.py─▶ <seed>/plots/summary.json (best_f1_threshold)
   ─retrain_on_trainval.py─▶ final_model.{ckpt,config.yaml,json}  # json: + descriptor_columns + desc_stats (Phase 2)
   ─predict.py─▶ <out>.csv (+ <out>_eval/ if --label-col)         # Phase 2: re-applies desc_stats from final_model.json
```
Descriptor columns are **positional** — everything after the first `id_column_count` (per `DATASET_REGISTRY`); `_split_ids_and_descs` makes the cut, `prepare_data.py` records the names in `manifest.json`. Also in [tech.md → Data lineage](../tech.md#data-lineage--raw--cache--model--prediction). Raw parquet = unnormalised source of truth; processed `data*.pt` = cached `Data` (desc raw now, normalised in Phase 2).

## Changes
- Added (2026-06-09, training path A–E):
  - `cfg.dataset.desc_dim`, `cfg.dataset.standardize_desc` ([dataset_config.py](../../graphgps/config/dataset_config.py)); `cfg.gnn.desc_proj_dim` ([custom_gnn_config.py](../../graphgps/config/custom_gnn_config.py)).
  - `LineGraphWithDescHead` (`@register_head('line_graph_with_desc')`) in [san_graph.py](../../graphgps/head/san_graph.py) — post-readout `f(desc)=Linear(desc_dim→desc_proj_dim)→act` concat before `out_layer`; `MOLECULAR DESCRIPTORS CONSUMED HERE` marker.
  - [tests/test_descriptor_head.py](../../tests/test_descriptor_head.py) (CPU, seconds).
  - [configs/biodegradability/Biodeg-DGT-Pipeline-WithDesc.yaml](../../configs/biodegradability/Biodeg-DGT-Pipeline-WithDesc.yaml).
- Changed (2026-06-09):
  - [biodeg.py](../../graphgps/loader/dataset/biodeg.py) + [biodeg_gwu.py](../../graphgps/loader/dataset/biodeg_gwu.py): `standardize_desc` ctor arg; separate processed cache (`data_stdesc.pt`); train-split z-score in `process()`; `desc_stats.json` persistence.
  - [master_loader.py](../../graphgps/loader/master_loader.py): pass `cfg.dataset.standardize_desc` to both loaders.
- Removed: none.
- **Deferred to post-dry-run:** group G (predict.py descriptor path), retrain→`final_model.json` lineage propagation, group F (docs). Reason: only testable after a WithDesc model is trained, and the manifest plumbing depends on the real on-disk dataset paths.

## Testing / Validation
- Tests run: `pytest tests/test_descriptor_head.py`, `pytest tests/test_dataset.py`, `pytest -m e2e`
- Manual checks: 3-epoch dry-run with `-WithDesc.yaml`; marker grep invariant
- Performance impact: head-only; negligible params over baseline
- Security/permissions considerations: none

## Risks & Rollback
- Risks: normalisation leakage if stats not train-only; cache staleness after standardisation change; predict.py parity drift.
- Mitigations: persist train-only stats; document `rm -rf processed/`; mirror stats in predict.py.
- Rollback plan: ablation toggle means `gnn.head: line_graph` fully restores baseline behaviour; revert branch if needed.

## Notes / Decisions made during implementation
- **2026-06-09 — Standardisation = Option A.** Normalise in the loader `process()` (z-score, train-split μ/σ), not a runtime transform.
- **2026-06-09 — Persist column NAMES + μ/σ** to a per-dataset `desc_stats.json`, and propagate both into `final_model.json` so prediction is self-contained and order-safe (descriptors matched by name, not position).
- **2026-06-09 — Separate processed cache for the desc model.** The normalised-desc data goes to its own cache; the baseline's raw-desc `processed/` is preserved (no `rm`). Implies the config selects which cache → YAML is *relevant* to data processing (see open item).
- **2026-06-09 — Backbone carry-through needs no code.** `batch.desc` `[B, desc_dim]` rides PyG collation; not routed via `to_dense_batch`; only the new head reads it (descriptors are global, not per-node/edge). Invariant enforced by the `MOLECULAR DESCRIPTORS` grep (N ENTER + 1 CONSUMED).
- **2026-06-09 — predict.py descriptor source (WithDesc).** Descriptors are supplied as **input-CSV columns** (not computed from SMILES). predict.py validates them against `final_model.json` → `descriptor_columns` (error on missing), **reorders by name to the training order** (order matters to the model), and applies `desc_stats`. See group G.
- **2026-06-09 — `f(desc)` = small MLP, `desc_proj_dim` tunable.** `Linear(desc_dim → gnn.desc_proj_dim) → GELU`, injected post-readout before `out_layer`. `desc_proj_dim` (default 128) is a YAML knob to modulate descriptor influence; model-only → no cache invalidation when swept. Confirmed "after the readout" matches the DGT author's guidance.
- **2026-06-09 — Minimal testing.** One cheap head unit test + one re-run of `test_dataset.py`; primary verification is the group-E 3-epoch dry-run, then the full 4-seed run. No per-step tests.
- **2026-06-09 — Descriptor-column SELECTION (gwu study side-task; pr-1 extension).** Added `dataset.desc_include` / `desc_exclude` / `desc_columns` (precedence: columns > include > all; exclude last); each non-empty selection auto-keys a separate processed cache `data_stdesc_<hash8>.pt`; head asserts `desc_dim` matches the selected width. New helper [_desc_select.py](../../graphgps/loader/dataset/_desc_select.py); loaders + master_loader updated; configs `*-WithDesc-gwu.yaml` / `-nongwu.yaml`; test [test_desc_select.py](../../tests/test_desc_select.py). Plus [scripts/select_features_from_shap.py](../../scripts/select_features_from_shap.py) (SHAP→`desc_columns` from S3). Touches only fork-added files + additive config (author core untouched). Project record: [projects/gwu.md](../projects/gwu.md).
- **2026-06-10 — gwu descriptor-type study COMPLETE (5/5).** Ordering by test AUC: non-GWU 0.9004 ± 0.0004 (+0.0183) > all 0.8966 > SHAP-selected-94 0.8864 > baseline 0.8821 > GWU-only 0.8728 (−0.0093). Conclusion: the RDKit (non-GWU) descriptors carry the gain; GWU/QM descriptors don't help. overview.md Phase 2/3/4 marked done; results in [projects/gwu.md](../projects/gwu.md).
- **2026-06-10 — Follow-up #6: `dim_hidden` exploration.** Try lower `gt.dim_hidden` (must equal `gnn.dim_inner`) — motivated by small descriptor sets (e.g. GWU-only = 40). Tracked in [projects/gwu.md](../projects/gwu.md) TODO #6.

## Group G — predict.py descriptors — DONE (2026-06-10)
- **Loader fix:** `desc_stats.json` is now **selection-keyed** (`desc_stats{suffix}.json`) so multiple subsets don't overwrite one stats file (was a latent collision bug).
- **retrain_on_trainval.py:** for `line_graph_with_desc` models, resolves the dataset's suffix-keyed `desc_stats` (via the same `_desc_select` keying) and embeds `descriptor_columns` + `desc_stats` (mean/std) + `desc_dim` into `final_model.json`.
- **predict.py:** if `gnn.head == line_graph_with_desc`, reads `descriptor_columns` + `desc_stats` from the manifest, validates the input CSV has those columns, **reorders by name to training order**, applies the persisted z-score, and attaches `data.desc`. Rows with missing/non-finite descriptors are skipped with a `remarks` note. SMILES-only models unchanged. Docstring updated.
- **Verify on remote:** after `retrain_on_trainval.py` on a WithDesc run, `final_model.json` has `descriptor_columns`/`desc_stats`; `predict.py` on a CSV containing those descriptor columns produces scores (and errors clearly if columns are missing).

## Group F — docs — DONE (2026-06-10)
- **tech.md:** Stage-4 readout documents the `line_graph_with_desc` variant (`f(desc)=Linear(desc_dim→desc_proj_dim)→GELU` concat with pooled `[atom ‖ bond]` before `out_layer`; descriptors never enter the backbone); Head row in the GraphGPS-integration table updated.
- **config_reference.md:** `gnn.head` comment updated ("(planned)" → implemented); new "Molecular descriptor fields (Phase 2)" subsection for `dataset.desc_dim` / `standardize_desc` / `desc_include`/`exclude`/`columns` + `gnn.desc_proj_dim`, with cache-invalidation notes.
- **trained_models.md:** new "Final model with molecular descriptor" section — non-GWU/RDKit winner (AUC 0.9004 ± 0.0004) + the predict.py deployment-bundle contract.
- **overview.md:** Phase 2/3/4 already marked done (prior session).

## Out of scope for PR-1 (tracked elsewhere)
- **Descriptors-only MLP sanity baseline** → moved to overview.md → Future work.
- **#6 `dim_hidden` / `desc_proj_dim` exploration** → [projects/gwu.md](../projects/gwu.md) TODO #6.
- **(optional)** repeat the descriptor-type study on biodeg (no-Reaxys).

## Open items to confirm
- ~~Separate-cache mechanism~~ — **RESOLVED 2026-06-09:** config flag `dataset.standardize_desc: True` (group A) makes the loader normalise + write a distinct `processed_file_names` (e.g. `data_stdesc.pt`), coexisting with the baseline's raw cache under one root. The new config is named with a `-WithDesc` suffix; the baseline config is untouched.
- ~~Per-model data-processing history record~~ — **RESOLVED 2026-06-09:** machine-readable lineage block added to `final_model.json` (source dataset, `descriptor_columns`, `desc_stats` μ/σ, `desc_dim`, prepare/train git SHAs); human-readable flow stays in [tech.md](../tech.md) + this log. No separate `documents/data_processing.md`.
- (next sub-decision) **`f(desc)`: raw concat vs small MLP** before concatenation — ADR open sub-decision #2, still to resolve.

## References
- Spec: [overview.md → Phase 2](../overview.md#phase-2--descriptor-plumbing-late-fusion)
- Baseline results: [trained_models.md](../trained_models.md)
- Related PRs: <links>
