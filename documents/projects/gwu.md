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
- [ ] **3. GWU-only** (desc_dim 40) — config ready; run 4-seed
- [ ] **4. non-GWU** (desc_dim 207) — config ready; run 4-seed
- [ ] **5. selected list** — define `desc_columns`, then run 4-seed

## Results (test AUC, 4-seed mean ± std)

| # | Variant | desc_dim | Test AUC | Test F1 | Test acc | Δ vs baseline | Notes |
|---|---|---|---|---|---|---|---|
| 1 | none | — | 0.8821 ± 0.0034 | 0.7836 | 0.7950 | 0 | baseline (params 1.252M) |
| 2 | all | 247 | 0.8966 ± 0.0027 | 0.8191 | 0.8242 | +0.0145 | descriptors help; best-val epoch 15; params 1.284M |
| 3 | gwu | 40 | | | | | |
| 4 | non-gwu | 207 | | | | | |
| 5 | selected | | | | | | |

Fill each row from `results/DGT/<config_basename>/agg/test/best.json` after the run.

### Full metrics (4-seed mean)

| No. | Variant | Accuracy | Precision | Recall | F1-Score | AUROC |
|---|---|---|---|---|---|---|
| 1 | none (baseline) | 0.7950 | 0.7936 | 0.7743 | 0.7836 | 0.8821 |
| 2 | all (247) | 0.8242 | 0.8094 | 0.8299 | 0.8191 | 0.8966 |
| 3 | GWU only (40) | | | | | |
| 4 | non-GWU (207) | | | | | |
| 5 | selected | | | | | |

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

### Variant 5 — selected list
Copy `...-WithDesc-gwu.yaml`, replace `desc_include: ['_gwu']` with
`desc_columns: [<exact names>]`, and set `desc_dim` to the list length. Each
distinct selection auto-keys its own processed cache (hash of the resolved
columns), so variants never collide.

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
