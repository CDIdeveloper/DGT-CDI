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
| 3 | GWU only | colnames ending `_gwu` | ~40 | TBD | yes (selection) |
| 4 | non-GWU | all except `_gwu` | ~207 | TBD | yes (selection) |
| 5 | selected list | explicit column list | N | TBD | yes (selection) |

## TODO
- [x] **1. baseline (no desc)** — AUC 0.8821 ± 0.0034 ([trained_models.md](../trained_models.md))
- [ ] **2. all descriptors** — config ready; run 4-seed (commands below)
- [ ] **3. GWU-only** — needs the descriptor-selection feature
- [ ] **4. non-GWU** — needs the descriptor-selection feature
- [ ] **5. selected list** — needs the descriptor-selection feature

## Results (test AUC, 4-seed mean ± std)

| # | Variant | desc_dim | Test AUC | Test F1 | Test acc | Δ vs baseline | Notes |
|---|---|---|---|---|---|---|---|
| 1 | none | — | 0.8821 ± 0.0034 | 0.7836 | 0.7950 | 0 | baseline |
| 2 | all | 247 | | | | | |
| 3 | gwu | ~40 | | | | | |
| 4 | non-gwu | ~207 | | | | | |
| 5 | selected | | | | | | |

Fill each row from `results/DGT/<config_basename>/agg/test/best.json` after the run.

## Commands

### Variant 2 — all descriptors (runnable now, no new code)
```bash
# 3-epoch dry-run (also builds datasets/biodeg_gwu/processed/data_stdesc.pt)
python main.py --cfg configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc.yaml \
  --repeat 1 seed 0 wandb.use False optim.max_epoch 3

# full 4-seed run
python main.py --cfg configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc.yaml \
  --repeat 4 seed 0 wandb.use False optim.max_epoch 50

# read result
cat results/DGT/Biodeg-GWU-DGT-Pipeline-WithDesc/agg/test/best.json
```

### Variants 3 / 4 / 5
Pending the **descriptor-selection feature** (design under discussion — see below). Once
landed, each variant is its own config (`...-WithDesc-gwu.yaml`, `...-nongwu.yaml`, etc.).

## Organization (proposed)
- **Pipeline code** (descriptor column selection) → `graphgps` loader + config (a small pr-1 extension; it's a reusable pipeline capability, not project-specific code → no separate `projects/` code folder).
- **Variant configs** → `configs/biodegradability/` (consistent with existing); group the selection variants by `-gwu`/`-nongwu`/`-sel` suffixes.
- **Records** → this file.
- **Results** → `results/DGT/<config_basename>/` (each variant = own config basename = own results dir; no collisions).

## Descriptor-selection design (3/4/5) — UNDER DISCUSSION
To be finalised before implementing. Open points: selection config shape
(`desc_include` / `desc_exclude` / `desc_columns`), per-selection processed-cache
key (avoid collisions on `data_stdesc.pt`), and `desc_dim` bookkeeping per variant.
