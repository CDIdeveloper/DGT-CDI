# Introduction

In this forked repo, plan to achieve two goals:
- introduce molecular descriptors into the model training and prediction.
- train and predict biodegradability using my own data.

# Introduction of DGT

> New to graph transformers? See [graph_transformer.md](graph_transformer.md) for a half-page primer (MPNN → graph transformer motivation, positional / structural encodings, local-vs-global hybrid).

DGT (Dual Graph Transformer) is a graph-transformer architecture for molecular property prediction. Its core idea is to view each molecule simultaneously as two coupled graphs and to fuse them in a single self-attention mechanism.

## Dual graph representation
- **Atom graph** $G^a = (N^a, E^a)$: atoms are nodes, bonds are directed edges.
- **Bond graph** $G^b = (N^b, E^b)$: obtained from the atom graph by a **line-graph transform**, so bonds become nodes and "two bonds share an atom" becomes the edge relation. Worked example in the paper: in phenyl formate, the ipso carbon (a single atom in $G^a$) corresponds to three edges sharing the same feature vector in $G^b$.

This dual view lets DGT encode comprehensive molecular information in a uniform way:

| Information | Where it lives |
|---|---|
| Atom features | Node vectors in $G^a$ / edge vectors in $G^b$ |
| Bond features | Node vectors in $G^b$ / edge vectors in $G^a$ |
| Topology / structure | Node–node pairwise matrices in both graphs |
| Ring structure | Ring structural encoding (RSE) added to pairwise matrices |
| Relative position | Shortest-path distance encoding (SPDE) + random-walk positional encoding (RWPE) |
| 3D info (optional, DGT(3D)) | Bond lengths → bond features; atom–atom distances → $E^a$; bond–bond angles → $E^b$ |

For 3D, interatomic distances are expanded with **enveloped Bessel basis** functions (smooth decay to zero at a cutoff, with vanishing first and second derivatives) and bond–bond angles are encoded with **normalised spherical harmonics**, giving a rotationally consistent geometric representation.

## Graph transformer layer
DGT extends biased multi-head self-attention so that pairwise node–node features act in two places at once:

$$
s_{i,j} \;\propto\; \exp\!\left(\frac{Q_i K_j^\top}{\sqrt{d_k}} + E^{\text{att}}_{i,j}\right), \qquad h_i \;=\; \sum_j s_{i,j}\bigl(V_j + E^{\text{val}}_{i,j}\bigr)
$$

- $E^{\text{att}}$ injects pairwise features (SPDE + RWPE + RSE, plus 3D terms when available) as an attention bias.
- $E^{\text{val}}$ modulates the value/message itself, enriching the aggregated representation.
- The same MHA operation is run on **both** the atom graph and the bond graph; atom and bond features are mutually fused into each other's attention scores.
- Each DGT layer wraps MHA in a residual MLP and an FFN, with **BatchNorm** for training stability. Only the node-feature matrix is updated layer-by-layer; the pairwise feature matrix is shared across all layers.

## Readout
After $L$ stacked DGT layers, global average pooling is applied separately to the atom and bond representations, the two pooled vectors are concatenated, and a small MLP with ReLU activations predicts the target property.

## Benchmarks and training
- **Datasets (MoleculeNet, 10 datasets / 58 sub-tasks):** BBBP, ClinTox, Tox21, SIDER (physiology); BACE, HIV (biophysics); ESOL, FreeSolv, Lipophilicity (physical chemistry); QM9 (quantum mechanics, 12 regression targets).
- **Splits:** scaffold split (80 / 10 / 10), repeated four times for averaging — testing generalisation to unseen scaffolds.
- **Metrics:** ROC-AUC (classification), RMSE (physical chemistry), MAE (QM9).
- **Pretraining (DGT(3D)):** on subsets of **PCQM4Mv2** (10K / 100K / 1M molecules) targeting the HOMO–LUMO gap, then fine-tuned on QM9. AdamW, batch size 128, lr 2e-4, cosine annealing with 5-epoch linear warmup, 100 epochs.
- **Baselines** compared against include: RF (ECFP4), Attentive FP, D-MPNN, pretrained GNN, GraphMVP, MolCLR, MoleBERT, GROVER, UniMol; and for 3D: SchNet, DimeNet, SphereNet, SE(3)-Transformer, 3DInfomax.

In this fork, configs for these benchmarks live in [configs/](../configs/), grouped by domain (`biophysics/`, `physical_chemistry/`, `physiology/`, `quantum_mechanics/`).

# Underlying framework: GraphGPS

The DGT codebase is built on **GraphGPS** ("General, Powerful, Scalable Graph Transformer", Rampášek et al., NeurIPS 2022) — confirmed in [readme.md](../readme.md): *"This implementation is developed from graphgps."* (upstream: https://github.com/rampasek/GraphGPS).

GraphGPS is a modular *recipe* and reference codebase for building graph transformers, not a single model. It sits on top of PyTorch Geometric's **GraphGym** experiment framework. It contributes three things:

1. **A hybrid block (MPNN + global attention).** Each *GraphGPS* layer runs a **local** message-passing module (GINE / GCN / GatedGCN / PNA) and a **global** attention module (Transformer / Performer / BigBird) *in parallel*, then sums them — combining local inductive bias with long-range mixing. **DGT does not use this layer** (see below).
2. **Positional / structural encodings as first-class input.** RWSE / LapPE / SignNet etc. are configurable encoders bolted onto node / edge features before the layers, rather than baked into the architecture.
3. **A GraphGym registry + YAML config system.** Layers, encoders, heads, losses, optimizers, and datasets are registered into named dictionaries via decorators (`@register_network`, `@register_layer`, …) and selected by string in a YAML config. The training loop, logging, multi-seed runs, and config merging come from GraphGym.

**What DGT actually takes from GraphGPS.** This repo is structurally a fork of the GraphGPS package layout — [graphgps/](../graphgps/) keeps the upstream subfolders (`act/`, `config/`, `encoder/`, `head/`, `layer/`, `loader/`, `loss/`, `network/`, `optimizer/`, `train/`, `transform/`) — but it only *reuses* points (2) and (3):

- **Adopted as-is** — the GraphGym harness (registry, training loop, config system) and the encoder infrastructure for positional / structural encodings.
- **Replaced** — the hybrid block (1) is **not** in DGT's data path. DGT registers its own `DGTLayer` (pure biased multi-head attention over a dual atom/bond graph, *no* MPNN side branch), `DGTModel`, and `line_graph` head as parallel alternatives. The upstream `GPSLayer`, `GPSModel`, SAN, and BigBird files remain in the repo as siblings / baselines, but DGT YAML configs never select them. (The only message-passing step in DGT is a single one performed once in the encoder — see [tech.md](tech.md#end-to-end-data-flow).)

See [tech.md](tech.md#integration-with-graphgps) for the full per-subfolder breakdown (which components are shared, parallel alternatives, or additive registrations).

# Roadmap

Two fork-specific goals (see top of file):
1. Plumb **molecular descriptors** into DGT training and prediction.
2. Train a DGT classifier on my **own biodegradability data** (SMILES + binary RD/NRD label + descriptors; train/test already split).

Design decisions taken up front:
- **Descriptor fusion:** late fusion at the head — `MLP([GAP(X^a) ‖ GAP(X^b) ‖ desc])`. Touches only the head, keeps the DGT backbone untouched, easy to ablate.
- **Splits:** honour the existing train/test; carve ~10 % of train as validation for early stopping. No scaffold re-split.
- **Data location:** `datasets/biodegradability/` — `train.csv`, `test.csv`, each with `smiles`, `label`, descriptor columns.

The plan is phased; each phase has a verification check, matching the "Goal-Driven Execution" guideline in [CLAUDE.md](../CLAUDE.md).

## Phase 0 — Environment & sanity check
- Build the conda env from [environment.yaml](../environment.yaml).
- Reproduce one shipped run (e.g. BBBP) end-to-end to confirm the codebase trains and logs cleanly.
- **Verify:** training loss decreases, ROC-AUC matches the paper's BBBP ballpark.

## Phase 1 — Dataset integration
- Add `datasets/biodegradability/{train.csv, test.csv}`.
- Implement `graphgps/loader/dataset/biodegradability.py` as a PyG `InMemoryDataset` (pattern: [chiral3d_molecule_net.py](../graphgps/loader/dataset/chiral3d_molecule_net.py) / [aqsol_molecules.py](../graphgps/loader/dataset/aqsol_molecules.py)).
  - Per molecule: SMILES → RDKit `Mol` → atom/bond features (reuse OGB / MoleculeNet featurisation), `data.y` ∈ {0, 1}, `data.desc` = standardised descriptor vector.
  - Persist a `train_mask` / `val_mask` / `test_mask` (or pre-computed split indices) reflecting the supplied split, with ~10 % of original train held out as validation.
  - Drop unparseable SMILES with a logged warning (don't silently skip).
- Register a `PyG-Biodegradability` format in [graphgps/loader/master_loader.py](../graphgps/loader/master_loader.py) and a matching `split_mode: predefined` branch in [graphgps/loader/split_generator.py](../graphgps/loader/split_generator.py).
- **Verify:** loader returns 3 DataLoaders, batch shapes look right, `batch.desc` is present and finite, label distribution printed.

## Phase 2 — Descriptor plumbing (late fusion)
- Standardise descriptors (z-score using train-set mean/std; persist stats so test/val use the same normalisation).
- Carry descriptors through the DGT backbone untouched (they live on `batch.desc` and survive `to_dense_batch` because they're a graph-level tensor).
- Add `DescriptorGraphHead` under [graphgps/head/](../graphgps/head/) — same as the current `line_graph` head but concatenates `batch.desc` (optionally passed through a small MLP) before the final MLP layer. Register via `head_dict`.
- **Verify:** unit test — forward pass with a toy batch returns the right output shape; ablation switch (`gnn.head: line_graph` vs `line_graph_with_desc`) toggles cleanly.

## Phase 3 — Config & first training run
- Create `configs/biodegradability/Biodeg-RWSE-SPDE-Rings.yaml`, modelled on [BBBP-RWSE-SPDE-Rings.yaml](../configs/physiology/BBBP-RWSE-SPDE-Rings.yaml):
  - `dataset.format: PyG-Biodegradability`, `task_type: classification_binary`, `split_mode: predefined`.
  - `gnn.head: line_graph_with_desc`, set `dataset.desc_dim` to the descriptor count.
  - Hyperparameters initially mirrored from BBBP (similar size, similar task); revisit after first run.
  - Loss: `cross_entropy`. If the train set is materially imbalanced (>~70/30), switch to `weighted_cross_entropy` or `focal_loss` (both already in [graphgps/loss/](../graphgps/loss/)).
- Run `python main.py --cfg configs/biodegradability/Biodeg-RWSE-SPDE-Rings.yaml --repeat 4 seed 0`.
- **Verify:** val ROC-AUC > random; test ROC-AUC is logged; no NaN losses; W&B run visible.

## Phase 4 — Ablation: does the descriptor channel help?
- Train three configs (4 seeds each, same data, same split):
  1. DGT only (`head: line_graph`).
  2. DGT + descriptors (`head: line_graph_with_desc`).
  3. Descriptors only (small MLP baseline) — quick sanity check for descriptor signal.
- Report mean ± std test ROC-AUC, AUPRC, accuracy at the optimal-F1 threshold.
- **Verify:** clear ordering and confidence intervals; decide whether to keep descriptors in the default config.

## Phase 5 — Interpretation (optional, paper-aligned)
- Implement Grad-SAM-style attention attribution (Methods §9 of the paper, [nc_paper_lean.md](paper/nc_paper_lean.md)) over the DGT attention maps.
- Pick a handful of RD vs NRD molecules, render atom-level importance overlays, sanity-check against known biodegradability substructures (e.g., ester / amide hydrolysis sites, halogenation patterns).
- **Verify:** importance maps are non-trivial (not all-uniform) and qualitatively reasonable.

## Open items / assumptions to confirm later
- Exact descriptor list and dimensionality (currently treated as a black-box vector of length `desc_dim`).
- Class balance of the RD/NRD labels — drives the loss choice in Phase 3.
- Whether we eventually want pretraining on a public biodegradability set (e.g. EPI BIOWIN or ECHA REACH) — not in the initial roadmap.

