# Trained models

Two tables — **HPO sweeps** (one per dataset × round; used to pick a winning config) and **Final models** (deployment-ready models produced by retraining on the winner). For the workflow that produces these entries, see [modeling_routine.md → Step 6 / Step 7](modeling_routine.md#step-6--iterate-hpo-across-configs-if-exploring-hyperparameters).

## HPO sweeps for models without molecular descriptor

One table per `(dataset, round)`. Each row is one variant; the **baseline** row sits at the top so every variant reads against a single reference. Each round changes one hyperparameter at a time from the current baseline (so a clear delta is attributable to that one change).

### biodeg_gwu — round 1

| Variant | Config | Change vs baseline | Test AUC (mean ± std, 4 seeds) | Test F1 (mean) | Test accuracy (mean) | Best-val epoch (median) | Δ Test AUC vs baseline | Notes |
|---|---|---|---|---|---|---|---|---|
| baseline | [Biodeg-GWU-DGT-Pipeline.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline.yaml) | — | 0.8821 ± 0.0034 | 0.7836 | 0.7950 | 31 | 0 | run date 2026-05-28; git SHA `<fill>` |
| L6 | [Biodeg-GWU-DGT-Pipeline-L6.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline-L6.yaml) | `gt.layers: 4 → 6` | 0.8755 ± 0.0079 | 0.7821 | 0.7933 | 27 | -0.0066 | regression; std 2.3× baseline. Extra depth does not help on biodeg_gwu. |
| dim256 | [Biodeg-GWU-DGT-Pipeline-dim256.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline-dim256.yaml) | `gt.dim_hidden / gnn.dim_inner: 128 → 256` | 0.8278 ± 0.0049 | 0.7203 | 0.7233 | 6 | -0.0543 | severe regression (~16× baseline std); peaked at epoch 6 then degraded. NaN/Inf in input tensor during training → instability with 4× params (4.97M). Drop width direction; if revisiting, lower LR + more warmup. |
| lr1e3 | [Biodeg-GWU-DGT-Pipeline-lr1e3.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline-lr1e3.yaml) | `optim.base_lr: 4e-4 → 1e-3` | 0.8250 ± 0.0082 | 0.7415 | 0.7375 | 8 | -0.0571 | severe regression (~17× baseline std); same NaN/Inf instability as dim256 (peaked at epoch 8, then degraded). Higher LR with the same model causes numerical issues; baseline LR=4e-4 was already well-tuned. |

All values readable directly from `results/DGT/<config_name>/agg/test/best.json` (no need to run `scripts/analyze_run.py` per seed for HPO comparison). Full fill-in commands + interpretation rule → [modeling_routine.md → Step 6](modeling_routine.md#step-6--iterate-hpo-across-configs-if-exploring-hyperparameters).

#### Round 1 verdict

All three Tier-1 variants regress vs baseline:

| Variant | Δ AUC | Clean comparison? |
|---|---|---|
| L6 (deeper) | −0.0066 (~2× baseline std) | yes — converges cleanly, just slightly worse |
| dim256 (wider) | −0.0543 (~16× baseline std) | **no** — NaN/Inf instability; peaked at epoch 6 then degraded |
| lr1e3 (higher LR) | −0.0571 (~17× baseline std) | **no** — same NaN/Inf instability; peaked at epoch 8 then degraded |

The two "no" cases are not fair comparisons — the instability invalidates the result. What we *can* conclude is structural:

1. **Depth does not help** on biodeg_gwu (L6 is the only clean comparison, modestly worse).
2. **Width + higher LR both push the architecture out of its stable regime** at the current `attn_dropout=0.3` and 10-epoch warmup. Either a smaller LR or longer warmup would likely fix the instability — but the baseline already converges cleanly at epoch 31, suggesting the architecture has found a sweet spot.
3. **Baseline (4 layers, dim=128, lr=4e-4) is the round-1 winner** and likely close to the architecture's ceiling for this dataset at this scale.

#### Proposed next step

I'd **skip a round 2 of HPO and go straight to Phase 2 (`DescriptorGraphHead`)** — the bigger expected lever for biodeg_gwu is whether the 247 RDKit/GWU descriptors add signal on top of the graph features, not finer-grained tuning of an architecture that's already in its sweet spot. Then revisit HPO *after* the descriptor channel is in place — the optimal hyperparameters for the descriptor-augmented model may differ from those of the graph-only baseline.

**If you do want a round 2 anyway** (e.g. to squeeze out a small gain on the graph-only baseline before Phase 2), the most plausible directions are **Tier-2** (modest expected delta but they avoid the instability pattern dim256/lr1e3 hit):

| Variant | Change vs baseline | Hypothesis |
|---|---|---|
| `dropout02` | `gt.dropout: 0.0 → 0.2` | small regularisation gain if there's any overfitting; tight baseline std suggests probably not |
| `attn_dropout05` | `gt.attn_dropout: 0.3 → 0.5` | stronger attention regularisation; complementary to `dropout02` |
| `wd1e3` | `optim.weight_decay: 1e-2 → 1e-3` | weaker WD; opposite hypothesis to the above (maybe baseline is over-regularised) |

These are cheap (same compute as round 1, same memory) but expected to give Δ AUC of 0.001–0.005 at most — within or just outside baseline std. Worth it only if you want a tight publication number; otherwise Phase 2 first.

### biodeg — round 1

Baseline-only (no HPO planned for this dataset yet — establishing a no-descriptor reference for the Phase-2 ablation, same as biodeg_gwu). Same architecture/hyperparameters as the biodeg_gwu baseline; the only deviation is `train.batch_size: 16` (32 OOMs on a 14.6 GiB T4 — biodeg has larger molecules; see [tech.md](tech.md#end-to-end-data-flow)).

| Variant | Config | Change vs baseline | Test AUC (mean ± std, 4 seeds) | Test F1 (mean) | Test accuracy (mean) | Best-val epoch (mean) | Δ Test AUC vs baseline | Notes |
|---|---|---|---|---|---|---|---|---|
| baseline | [Biodeg-DGT-Pipeline.yaml](../configs/biodegradability/Biodeg-DGT-Pipeline.yaml) | — | 0.9007 ± 0.0024 | 0.7918 | 0.8335 | 33 | 0 | run date 2026-06-08; git SHA `c1c5e1a`. precision 0.7699, recall 0.8160. batch_size=16 (T4 OOM at 32). Data: 8054 mols, ~39–41% pos, desc_dim=216. |

**Cross-dataset note (not a clean single-knob comparison — different data + smaller batch):** biodeg's baseline AUC (0.9007 ± 0.0024) is **+0.0186** over the biodeg_gwu baseline (0.8821 ± 0.0034) — biodeg appears to be a modestly easier/cleaner task at this scale. Tighter std too (0.0024 vs 0.0034). Healthy result; no `loss_fun` change was needed (class balance ~39–41% positive across all splits).

## Final model without molecular descriptor

One row per deployment-ready model — the output of running `scripts/retrain_on_trainval.py` on an HPO winner. Test metrics carry over from the **original 4-seed dgt-mode run** of the same config (the retrain has no held-out test estimate of its own; the original aggregate is the closest unbiased proxy — see [modeling_routine.md Step 5 → "Two senses of 'test data is used'"](modeling_routine.md#two-senses-of-test-data-is-used)).

All five test metrics below are the **4-seed mean** from `agg/test/best.json` (default 0.5 threshold). The optimal-F1 threshold is recorded separately for deployment-time use — apply it via `predict.py --threshold optimal-f1` or by re-thresholding the raw scores.

| Model name | Dataset | Winning config | Retrain mode | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUROC (mean ± std) | Optimal F1 threshold | Best-val epoch | Cloud bundle URI | Git SHA | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _(none yet)_ | | | | | | | | | | | | | |

How to fill this table + upload the bundle to cloud → [modeling_routine.md → Step 7](modeling_routine.md#step-7--retrain-winning-config-record-final-model-ship-deployment-bundle).
