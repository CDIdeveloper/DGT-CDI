# ADR 0001: Late-fusion of molecular descriptors via a dedicated readout head

## Status
Proposed  <!-- DRAFT — for iteration -->

## Date
2026-06-09

## Context
Fork goal #1 (see [overview.md](../overview.md)) is to feed precomputed **molecular descriptors** into DGT alongside the graph. The loaders already attach them: each `Data` carries `desc` of shape `[1, desc_dim]` (216 for `biodeg`, 247 for `biodeg_gwu`), marked with the `MOLECULAR DESCRIPTORS ENTER HERE` comments in [biodeg.py](../../graphgps/loader/dataset/biodeg.py) / [biodeg_gwu.py](../../graphgps/loader/dataset/biodeg_gwu.py). What's undecided is **how** the descriptor vector should enter the model.

Forces:
- Descriptors are **graph-level** (one vector per molecule), not per-atom/per-bond.
- We want a **clean ablation** ("does the descriptor channel help?", Phase 4) — ideally a one-line config toggle with the graph backbone byte-for-byte unchanged between arms.
- Minimal, surgical code; reuse the existing `line_graph` pooling head where possible.
- Descriptors are on heterogeneous raw scales → need **normalisation** with no train→val/test leakage.

## Decision
Adopt **late fusion at the readout head.**

1. **New head `line_graph_with_desc`** (under [graphgps/head/](../../graphgps/head/)) — a copy of `LineGraphHead` ([san_graph.py:57](../../graphgps/head/san_graph.py#L57)) that concatenates `batch.desc` (passed through a small descriptor MLP) with the pooled atom/bond embeddings before the final `out_layer`:
   `logits = MLP( GAP(Xᵃ) ‖ GAP(Xᵇ) ‖ f(desc) )`.
2. **Backbone untouched.** `batch.desc` is a graph-level tensor `[B, desc_dim]` from PyG collation; it does **not** pass through `to_dense_batch`, so encoders / DGT attention / pairwise tensors never see it. The graph-only and graph+desc arms share an identical backbone.
3. **Toggle via config** — `gnn.head: line_graph` (baseline) vs `line_graph_with_desc` (fusion). Register the head with `@register_head('line_graph_with_desc')`.
4. **Standardise descriptors** (z-score) using **train-split statistics only**, persisted so val/test/predict reuse identical normalisation.
5. **`desc_dim` is a config field** (`cfg.dataset.desc_dim`) so the head can size its MLP; GraphGym rejects unknown YAML keys, so the field must be registered first.
6. **Marker discipline.** Add a `MOLECULAR DESCRIPTORS CONSUMED HERE` comment at the `batch.desc` read site, so `grep -rn 'MOLECULAR DESCRIPTORS'` returns exactly N ENTER (one per dataset that carries desc) + one CONSUMED per fusion head — any other hit = a leak through the backbone.

## Alternatives considered
1. **Early fusion (append `desc` to atom node features before the encoder)** — pros: descriptors could influence attention/message passing. Cons: `desc` is graph-level, so it must be broadcast to every atom; pollutes the backbone (breaks "backbone untouched"); makes ablation messy (the two arms no longer share a backbone); inflates `dim_in`.
2. **Mid fusion (inject `desc` as an attention bias or a virtual global node)** — most expressive. Cons: most invasive (encoder + layer code), hardest to ablate, high complexity for uncertain gain. Reserve as an escalation only if late fusion plateaus.
3. **Descriptor-only MLP (no graph)** — not a fusion at all, but a useful **sanity baseline** to confirm the descriptors carry signal on their own. Keep it as a *separate* Phase-4 baseline, not the main mechanism.
4. **Late fusion at the head (chosen)** — minimal code, backbone untouched, clean one-line ablation toggle.

## Consequences
### Positive
- Graph backbone is identical across ablation arms → the descriptor delta is cleanly attributable.
- Smallest possible code surface: one new head + one config field + normalisation in the loader.
- Reuses the existing pooling head; descriptors stored once per molecule.

### Negative / Trade-offs
- Descriptors influence **only the final readout**, not message passing or attention — late fusion may underuse them, and strong graph features can dominate the concatenation. (Accepted: if the ablation shows the channel helps but is weak, mid-fusion is the documented escalation.)
- Standardisation introduces persisted state (train-set mean/std) that must stay in sync across loader, val/test, and `predict.py` (which keeps an intentional standalone copy of featurisation — parity must be maintained by hand).
- Changing standardisation invalidates the PyG `processed/` cache → `rm -rf datasets/<name>/processed/` required.

### Follow-ups
- [ ] Register `cfg.dataset.desc_dim` (see log TODO).
- [ ] Decide & implement standardisation location + stat persistence (open sub-decision below).
- [ ] Implement `DescriptorGraphHead`; add tests; create `-WithDesc` config; dry-run.
- [ ] Phase 4 ablation (graph-only vs graph+desc vs desc-only MLP).

## Open sub-decisions (to iterate before/while implementing)
- **Where to standardise — RESOLVED (Option A, 2026-06-09):** in the loader `process()`. Compute z-score μ/σ from the **train split only**, apply to all splits, and persist μ/σ **+ descriptor column names** to a per-dataset `desc_stats.json`; propagate both into `final_model.json` for leak-free, order-safe inference. The normalised data is written to a **separate processed cache** so the baseline's raw-desc cache is preserved (new model → new cache); which cache a run uses is config-driven (see log open item on the flag mechanism).
- **`f(desc)`: raw concat vs small MLP — RESOLVED (small MLP, 2026-06-09):** `f(desc) = Linear(desc_dim → desc_proj_dim) → GELU`. The projection dim is a **tunable YAML knob** `cfg.gnn.desc_proj_dim` (default 128), so the descriptor channel's width/influence can be modulated per run (model-only → no cache invalidation when swept) and it absorbs the fact that `desc_dim` varies by dataset (216 vs 247). Injection is **post-readout**: concat `f(desc)` with the pooled `[atom ‖ bond]` vector right before `out_layer` — matches the DGT author's "add the descriptors after the readout." NB: in `LineGraphHead` the per-node FC layers run *pre*-pool and the only post-pool transform is the `out_layer` linear; the descriptor enters at that post-pool stage.
- **How the head learns `desc_dim`:** RESOLVED — from `cfg.dataset.desc_dim` (explicit; sizes the descriptor MLP input).

## Links
- Implemented in PR(s): #1 (branch `mol-desc`)
- Related log(s): [../log/pr-1-mol-desc.md](../log/pr-1-mol-desc.md)
- Related ADRs: none yet
- Spec: [overview.md → Phase 2](../overview.md#phase-2--descriptor-plumbing-late-fusion) / [Phase 3](../overview.md#phase-3--config--first-training-run)
