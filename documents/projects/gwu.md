# Project: GWU biodegradability — molecular-descriptor-type study

> **Goal:** compare DGT performance on **biodeg_gwu** across different *types* of
> molecular descriptors, fused at the head (late fusion, `line_graph_with_desc`).
> An *application* of the Phase-2 descriptor pipeline ([pr-1 log](../log/pr-1-mol-desc.md),
> [ADR 0001](../adr/0001-pr-mol-desc.md)) — not pipeline development.

**Dataset:** biodeg_gwu (GWU batch 2). All-descriptor `desc_dim = 247`; ~43% positive.
**Controlled:** backbone + optimizer identical across variants (the biodeg_gwu baseline);
only the descriptor channel changes, so AUC deltas are attributable to the descriptor set.

## Variants

| # | Variant | Descriptors | desc_dim | Config | New code? |
|---|---|---|---|---|---|
| 1 | no descriptors (baseline) | none | — | `Biodeg-GWU-DGT-Pipeline.yaml` | done (Phase 1) |
| 2 | all descriptors | all | 247 | `Biodeg-GWU-DGT-Pipeline-WithDesc.yaml` | **no — runnable now** |
| 3 | GWU only | colnames containing `_gwu` | 40 | `Biodeg-GWU-DGT-Pipeline-WithDesc-gwu.yaml` | done (selection) |
| 4 | non-GWU | all except `_gwu` | 207 | `Biodeg-GWU-DGT-Pipeline-WithDesc-nongwu.yaml` | done (selection) |
| 5 | selected list | explicit column list | N | copy variant 3, set `desc_columns: [...]` | done (selection) |

> Descriptor counts (from the count command below): total **247** = **40** GWU (`_gwu`) + **207** non-GWU. `desc_dim` is set accordingly in the variant 3/4 configs (the head asserts it matches the selected width).

## TODO
- [x] **1. baseline (no desc)** — AUC 0.8821 ± 0.0034 ([trained_models.md](../trained_models.md))
- [x] **2. all descriptors** — AUC 0.8966 ± 0.0027 (+0.0145 vs baseline; F1 0.819, acc 0.824)
- [x] **3. GWU-only** (40) — AUC 0.8728 ± 0.0070 (−0.0093 vs baseline; QM descriptors don't help)
- [x] **4. non-GWU** (207) — AUC 0.9004 ± 0.0004 (+0.0183 vs baseline; **best variant**)
- [x] **5. selected** (94, SHAP-screened) — AUC 0.8864 ± 0.0055 (+0.0043; < non-GWU)
- [x] **6. desc_proj_dim sweep** (variant 3, GWU-only): {16, 32, 64} vs 128 → AUC ~0.873–0.876, all below baseline; 64 marginally best (0.8762). Narrowing the descriptor projection does **not** rescue the GWU/QM set (table below).

## Results (test AUC, 4-seed mean ± std)

| # | Variant | desc_dim | Test AUC | Test F1 | Test acc | Δ vs baseline | Notes |
|---|---|---|---|---|---|---|---|
| 1 | none | — | 0.8821 ± 0.0034 | 0.7836 | 0.7950 | 0 | baseline (params 1.252M) |
| 2 | all | 247 | 0.8966 ± 0.0027 | 0.8191 | 0.8242 | +0.0145 | descriptors help; best-val epoch 15; params 1.284M |
| 3 | gwu | 40 | 0.8728 ± 0.0070 | 0.7809 | 0.7892 | −0.0093 | QM descriptors alone HURT (below baseline); epoch 28; params 1.258M |
| 4 | non-gwu | 207 | 0.9004 ± 0.0004 | 0.8211 | 0.8267 | +0.0183 | **best variant**; RDKit drives the gain; tiny std; epoch 15; params 1.279M |
| 5 | selected | 94 | 0.8864 ± 0.0055 | 0.8093 | 0.8150 | +0.0043 | SHAP-screened (94 cols); helps but < non-gwu; epoch 17; params 1.265M |

NOTE selected means selected descriptors by SHAP values based 90% coverage.

Fill each row from `results/DGT/<config_basename>/agg/test/best.json` after the run.

### Full metrics (4-seed mean)

| No. | Variant | Accuracy | Precision | Recall | F1-Score | AUROC |
|---|---|---|---|---|---|---|
| 1 | none (baseline) | 0.7950 | 0.7936 | 0.7743 | 0.7836 | 0.8821 |
| 2 | all (247) | 0.8242 | 0.8094 | 0.8299 | 0.8191 | 0.8966 |
| 3 | GWU only (40) | 0.7892 | 0.7799 | 0.7830 | 0.7809 | 0.8728 |
| 4 | non-GWU (207) | 0.8267 | 0.8146 | 0.8281 | 0.8211 | 0.9004 |
| 5 | selected (94) | 0.8150 | 0.8015 | 0.8177 | 0.8093 | 0.8864 |


## Findings (5/5 variants complete)

Test-AUC ranking: **non-GWU (0.9004) > all (0.8966) > selected-94 (0.8864) > baseline (0.8821) > GWU-only (0.8728)**.

1. **The descriptor gain is entirely from the non-GWU (RDKit / functional-group) descriptors.** Non-GWU alone (+0.0183) beats *all* descriptors (+0.0145) with fewer features (207 vs 247) and a far tighter std (0.0004 vs 0.0027).
2. **The GWU/QM descriptors do not help here — they hurt.** GWU-only (40) lands *below* baseline (−0.0093), and adding GWU on top of RDKit dilutes the result (all < non-GWU). So the 40 QM descriptors carry little biodegradability signal for the DGT late-fusion head and add noise.
3. **SHAP-screened (94) helps modestly (+0.0043) but does not beat simply dropping GWU.** Consistent with (2): the SHAP screen (computed on the *analysis* model, over the qm_rdkit set) still retained QM descriptors the DGT head doesn't benefit from.
4. **Recommendation:** for a deployable biodeg_gwu model, use the **non-GWU (RDKit) descriptor set** — best AUC, tightest variance, fewer features.

**Caveat:** the SHAP ranking came from the analysis model, not DGT, so the selected-subset result (variant 5) is a heuristic. A DGT-native attribution (Grad-SAM / SHAP on the desc head) could screen differently — worth a follow-up if variant 5 is pursued further.

## `desc_proj_dim` sweep (variant 3, GWU-only, 40 descriptors)

Does narrowing the descriptor projection `f(desc)=Linear(40→desc_proj_dim)` help the small GWU set? **No** — AUC stays ~0.873–0.876 across `desc_proj_dim ∈ {16, 32, 64, 128}`, all **below baseline (0.8821)** and within each other's noise.

| desc_proj_dim | Test AUC | Test F1 | Test acc | best-val epoch | params |
|---|---|---|---|---|---|
| 16 | 0.8731 ± 0.0085 | 0.7809 | 0.7883 | 21 | 1.253M |
| 32 | 0.8731 ± 0.0104 | 0.7832 | 0.7892 | 38 | 1.254M |
| 64 | 0.8762 ± 0.0052 | 0.7836 | 0.7925 | 43 | 1.255M |
| 128 (variant 3) | 0.8728 ± 0.0070 | 0.7809 | 0.7892 | 28 | 1.258M |

### Full metrics (desc_proj_dim sweep, 4-seed mean)

| desc_proj_dim | Accuracy | Precision | Recall | F1-Score | AUROC |
|---|---|---|---|---|---|
| 16 | 0.7883 | 0.7763 | 0.7865 | 0.7809 | 0.8731 |
| 32 | 0.7892 | 0.7754 | 0.7934 | 0.7832 | 0.8731 |
| 64 | 0.7925 | 0.7846 | 0.7830 | 0.7836 | 0.8762 |
| 128 (variant 3) | 0.7892 | 0.7799 | 0.7830 | 0.7809 | 0.8728 |

`desc_proj_dim=64` is marginally best but within noise; none reach baseline. **Takeaway:** the GWU/QM descriptors lack useful biodegradability signal for the DGT head regardless of projection width — it's a *signal* problem, not a capacity/projection one. (Reinforces the main finding: non-GWU/RDKit descriptors carry the gain.)

## Commands

### Variant 2 — all descriptors (runnable now, no new code)
```bash
# 3-epoch dry-run (also builds datasets/biodeg_gwu/processed/data_stdesc.pt)
python main.py --cfg configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc.yaml \
  --repeat 1 seed 0 wandb.use False optim.max_epoch 3

# remove modeling results from dry-run (optional but recommended)
rm -rf results/DGT/Biodeg-GWU-DGT-Pipeline-WithDesc/

# full 4-seed run
python main.py --cfg configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc.yaml \
  --repeat 4 seed 0 wandb.use False optim.max_epoch 50

# read result
cat results/DGT/Biodeg-GWU-DGT-Pipeline-WithDesc/agg/test/best.json
```

### Descriptor count (run first, sets desc_dim for variants 3/4)
```bash
python - <<'PY'
import json, sys
sys.path.insert(0, 'graphgps/loader/dataset')
from _desc_select import select_descriptor_columns
cols = json.load(open('datasets/biodeg_gwu/raw/manifest.json'))['descriptor_columns']
print('total   :', len(cols))
print('gwu     :', len(select_descriptor_columns(cols, include=['_gwu'])))
print('non-gwu :', len(select_descriptor_columns(cols, exclude=['_gwu'])))
PY
```
Put `gwu` → `desc_dim` in the `-gwu.yaml`, `non-gwu` → `desc_dim` in the `-nongwu.yaml`.

### Variant 3 — GWU only
```bash
# after setting dataset.desc_dim in the config:
python main.py --cfg configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc-gwu.yaml \
  --repeat 1 seed 0 wandb.use False optim.max_epoch 3     # dry-run (builds data_stdesc_<hash>.pt)
rm -rf results/DGT/Biodeg-GWU-DGT-Pipeline-WithDesc-gwu/  # optional, before 4-seed
python main.py --cfg configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc-gwu.yaml \
  --repeat 4 seed 0 wandb.use False optim.max_epoch 50
```

### Variant 4 — non-GWU
```bash
python main.py --cfg configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc-nongwu.yaml \
  --repeat 4 seed 0 wandb.use False optim.max_epoch 50    # (dry-run first as above)
```

### Variant 5 — selected list (SHAP-screened)
Generate `desc_columns` from a SHAP-ranking CSV with
[scripts/select_features_from_shap.py](../../scripts/select_features_from_shap.py)
(reads the CSV from **S3**; criteria `--cumulative` / `--top-k` / `--abs` / `--coef-thresh`):
```bash
# recommended: cumulative coverage (use the qm_rdkit ranking = all features)
python scripts/select_features_from_shap.py \
  --s3-key ts_project_1/data/biodegradation/features/GWU/feature_rank_shap_gwu_b2_qm_rdkit.csv \
  --trans-learn-path /home/jovyan/tools/trans_learn \
  --cumulative 0.90
```
Paste the printed `desc_dim` + `desc_columns` into a copy of `...-WithDesc-gwu.yaml`
(remove `desc_include`/`desc_exclude`). Repeat as Variant 3 as above. Each distinct selection auto-keys its own
processed cache (hash of the resolved columns), so variants never collide.
(For reference, on the **QM-only** ranking — 40 feats — cumulative 0.90→24, 0.95→28; the qm_rdkit ranking covers all ~247 and will give different counts — the script prints the count + coverage.)
**SHAP caveats:** importance is model-specific (analysis model ≠ DGT) and must be
train-only — see the design section below.

## Organization (proposed)
- **Pipeline code** (descriptor column selection) → `graphgps` loader + config (a small pr-1 extension; it's a reusable pipeline capability, not project-specific code → no separate `projects/` code folder).
- **Variant configs** → `configs/biodegradability/` (consistent with existing); group the selection variants by `-gwu`/`-nongwu`/`-sel` suffixes.
- **Records** → this file.
- **Results** → `results/DGT/<config_basename>/` (each variant = own config basename = own results dir; no collisions).

## Descriptor-selection design (3/4/5) — IMPLEMENTED (2026-06-09)
- **Config:** `dataset.desc_include` / `desc_exclude` (substring match) / `desc_columns` (explicit). Precedence: columns > include > all; exclude applied last. Registered in [dataset_config.py](../../graphgps/config/dataset_config.py).
- **Cache key:** each non-empty selection auto-keys its own processed cache `data_stdesc_<hash8>.pt` (hash of the resolved, ordered column list) — no collisions with the all-descriptor `data_stdesc.pt` or between subsets. Logic in [_desc_select.py](../../graphgps/loader/dataset/_desc_select.py).
- **desc_dim:** explicit `dataset.desc_dim` per config; the `line_graph_with_desc` head asserts it equals the actual (selected) descriptor width → wrong value fails loudly.
- Loaders [biodeg.py](../../graphgps/loader/dataset/biodeg.py) / [biodeg_gwu.py](../../graphgps/loader/dataset/biodeg_gwu.py) apply the selection in `process()`; selected columns + stats persisted to `desc_stats.json`. Pure-logic test: [tests/test_desc_select.py](../../tests/test_desc_select.py).
