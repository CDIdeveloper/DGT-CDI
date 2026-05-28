# Trained models for biodeg_gwu

Two tables — **HPO sweeps** (one per dataset × round; used to pick a winning config) and **Final models** (deployment-ready models produced by retraining on the winner). For the workflow that produces these entries, see [modeling_routine.md → Step 6 / Step 7](modeling_routine.md#step-6--iterate-hpo-across-configs-if-exploring-hyperparameters).

## HPO sweeps for models without molecular descriptor

One table per `(dataset, round)`. Each row is one variant; the **baseline** row sits at the top so every variant reads against a single reference. Each round changes one hyperparameter at a time from the current baseline (so a clear delta is attributable to that one change).

### biodeg_gwu — round 1

| Variant | Config | Change vs baseline | Test AUC (mean ± std, 4 seeds) | Test F1 (mean) | Test accuracy (mean) | Best-val epoch (median) | Δ Test AUC vs baseline | Notes |
|---|---|---|---|---|---|---|---|---|
| baseline | [Biodeg-GWU-DGT-Pipeline.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline.yaml) | — | 0.8821 ± 0.0034 | 0.7836 | 0.7950 | 31 | 0 | run date 2026-05-28; git SHA `<fill>` |
| L6 | [Biodeg-GWU-DGT-Pipeline-L6.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline-L6.yaml) | `gt.layers: 4 → 6` | <fill> | <fill> | <fill> | <fill> | <fill> | matches paper's BBBP recipe |
| dim256 | [Biodeg-GWU-DGT-Pipeline-dim256.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline-dim256.yaml) | `gt.dim_hidden / gnn.dim_inner: 128 → 256` | <fill> | <fill> | <fill> | <fill> | <fill> | wider model |
| lr1e3 | [Biodeg-GWU-DGT-Pipeline-lr1e3.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline-lr1e3.yaml) | `optim.base_lr: 4e-4 → 1e-3` | <fill> | <fill> | <fill> | <fill> | <fill> | aggressive LR |

All values readable directly from `results/DGT/<config_name>/agg/test/best.json` (no need to run `scripts/analyze_run.py` per seed for HPO comparison). Full fill-in commands + interpretation rule → [modeling_routine.md → Step 6](modeling_routine.md#step-6--iterate-hpo-across-configs-if-exploring-hyperparameters).

## Final model without molecular descriptor

One row per deployment-ready model — the output of running `scripts/retrain_on_trainval.py` on an HPO winner. Test metrics carry over from the **original 4-seed dgt-mode run** of the same config (the retrain has no held-out test estimate of its own; the original aggregate is the closest unbiased proxy — see [modeling_routine.md Step 5 → "Two senses of 'test data is used'"](modeling_routine.md#two-senses-of-test-data-is-used)).

| Model name | Dataset | Winning config | Retrain mode | Test AUC (4-seed mean ± std) | Test AUPRC | Test F1 (at optimal threshold) | Test accuracy | Optimal F1 threshold | Best-val epoch | Cloud bundle URI | Git SHA | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _(none yet)_ | | | | | | | | | | | | |

How to fill this table + upload the bundle to cloud → [modeling_routine.md → Step 7](modeling_routine.md#step-7--retrain-winning-config-record-final-model-ship-deployment-bundle).
