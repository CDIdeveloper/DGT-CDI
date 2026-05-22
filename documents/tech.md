# Introduction

Tech inventory for this DGT fork, grounded in both the codebase and the methods section of the paper. For background concepts (what a graph transformer is, why PE/SE encodings matter), see [graph_transformer.md](graph_transformer.md).

## Framework & core stack
- **PyTorch 2.1** (CUDA 12.1) as the DL backend.
- **PyTorch Geometric (PyG) 2.0.4** for graph data, batching, datasets, plus the extension wheels (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`, `pyg-lib`).
- **GraphGym / GraphGPS** as the experiment harness ([main.py](../main.py) is a thin wrapper). The [graphgps/](../graphgps/) package extends it with DGT-specific layers, networks, encoders, heads, transforms, losses, and optimizers via the GraphGym registry.
- **OGB 1.3.6** for OGB datasets (used for **PCQM4Mv2** pretraining as described in the paper).
- **RDKit 2025.9.1** for SMILES parsing and molecular preprocessing.
- **torchmetrics 1.2** and **libauc 1.1** for evaluation metrics (ROC-AUC, RMSE, MAE, AUPRC).

## Integration with GraphGPS

See [overview.md](overview.md#underlying-framework-graphgps) for the conceptual intro. This subsection lists what DGT contributes in each subfolder of [graphgps/](../graphgps/) and, importantly, how those contributions relate to the upstream components.

**Three modes of integration.** Not every DGT contribution "extends" GraphGPS. Mapping each row of the table below to one of these clarifies the relationship:

- *Shared / reused* — DGT uses the upstream component as-is. Mostly the **harness**: GraphGym registry, training loop, config-merging, encoder / optimizer / train infrastructure.
- *Parallel alternative* — DGT registers its own object next to an upstream one, and DGT YAML configs select the DGT version. Upstream stays in the repo as a sibling / baseline but is not in DGT's data path. Applies to **Layer**, **Network**, **Head**, and some **Transforms**.
- *Additive registration* — DGT adds new entries to a registry dictionary that upstream already populates. The upstream entries remain usable. Applies to **Encoders**, **Loss**, **Loader**, and **Config schema**.

The upstream `GPSLayer` ([graphgps/layer/gps_layer.py](../graphgps/layer/gps_layer.py), the "local MPNN ‖ global attention" hybrid) is a *parallel alternative* case: still present, never invoked by DGT configs (see [End-to-end data flow](#end-to-end-data-flow) below).

Anchor points:

- **Same harness.** [main.py](../main.py) is essentially GraphGPS's entry point — it calls `set_cfg`, `load_cfg`, `create_loader`, `create_model`, `create_optimizer`, `create_scheduler`, `train` from `torch_geometric.graphgym`. Only the `cfg.device` selection and `pretrained.*` hooks are customised.
- **Same package skeleton.** Identical subfolders under [graphgps/](../graphgps/). [graphgps/__init__.py](../graphgps/__init__.py) imports each so decorators register on import.
- **Original GPS pieces kept as siblings.** The upstream layer and model are still present: [graphgps/layer/gps_layer.py](../graphgps/layer/gps_layer.py), [graphgps/network/gps_model.py](../graphgps/network/gps_model.py), plus baselines (`san_layer`, `san2_layer`, `bigbird_layer`, `san_transformer`, `big_bird`, `custom_gnn`). DGT is added **next to them**, not on top of them.

| Framework layer | Mode | Upstream GraphGPS provides | DGT contribution in this repo |
|---|---|---|---|
| **Layer** (per-block compute) | parallel alternative | `GPSLayer` (local-MPNN ‖ global-attn block) | [dgt_layer.py](../graphgps/layer/dgt_layer.py): `DGTLayer`, `NodeGTLayer`, `EdgeGTLayer` — biased multi-head attention with pairwise $E^{\text{att}}$ / $E^{\text{val}}$, run on both atom and bond graphs. **DGT configs use `DGTLayer`, not `GPSLayer`.** |
| **Network** (model wiring) | parallel alternative | `GPSModel` | [dgt_model.py](../graphgps/network/dgt_model.py): `DGTModel`, `DGTModel3D` registered via `@register_network('DGTModel'…)`; adds `NodeEdgeEncoder` (atom↔bond mutual fusion) and `PositionEncoder` (Bessel + spherical harmonics for 3D) |
| **Encoders** (node / edge / PE) | additive registration | RWSE, LapPE, SignNet, linear node/edge | [topology_edge_encoder.py](../graphgps/encoder/topology_edge_encoder.py) for SPDE, [relative_pe_encoder.py](../graphgps/encoder/relative_pe_encoder.py) for pairwise positional features, composed encoders (`LinearEdge+RWSE-SPDE`) registered alongside the originals |
| **Transforms** (pre-compute) | additive registration | RWSE / Laplacian PE precomputation | `add_rings` (RSE), `compute_shortest_paths` (SPDE), `line_graph` (atom→bond graph) in [transform/transforms.py](../graphgps/transform/transforms.py) |
| **Head** (readout) | parallel alternative | `san_graph`, OGB heads | `line_graph` head — pools atom and bond representations separately, concatenates, then MLP — in [graphgps/head/](../graphgps/head/) |
| **Loader** | additive registration | OGB, PyG datasets, ZINC, etc. | `chiral3d_molecule_net`, `aqsol_molecules` added under [graphgps/loader/dataset/](../graphgps/loader/dataset/); QM9 scaffold splits under [datasets/QM9_split/](../datasets/QM9_split/) |
| **Loss** | additive registration | Standard CE / MSE | Multilabel focal / weighted variants in [graphgps/loss/](../graphgps/loss/) |
| **Config schema** | additive registration | `cfg.gt.*`, `cfg.posenc_*` | Extra fields under `cfg.dataset.spd`, `cfg.dataset.rings`, `cfg.dataset.spd_max_length`, etc., defined in [graphgps/config/](../graphgps/config/) |
| **Pretrained-weight loading** | new | — | [graphgps/finetuning.py](../graphgps/finetuning.py) for the DGT(3D) PCQM4Mv2 → QM9 transfer |
| **Harness** (training loop, registry, config merging, GraphGym wrappers) | shared / reused | full GraphGym + GraphGPS infrastructure | unchanged |

Selection is wired through YAML: e.g. [configs/physiology/BBBP-RWSE-SPDE-Rings.yaml](../configs/physiology/BBBP-RWSE-SPDE-Rings.yaml) sets `model.type: DGTModel`, `gt.layer_type: None+DGT`, `dataset.node_encoder_name: LinearNode`, `dataset.edge_encoder_name: LinearEdge+RWSE-SPDE`, `gnn.head: line_graph`. Every one of those strings is a registry key resolved at startup. To swap in the upstream GPS layer instead, you would change `model.type: GPSModel` and `gt.layer_type` to a GraphGPS string such as `GINE+Transformer`.

**Bottom line.** GraphGPS supplies the *harness* — GraphGym registry, training loop, config system, encoder/loss/loader infrastructure — and DGT reuses it. For the **model itself** (layer, network, head, key transforms) DGT does *not* extend the upstream GPSLayer; it registers a parallel `DGTLayer` / `DGTModel` / `line_graph` head that the DGT YAML configs select instead. The original `GPSLayer` / `GPSModel` files remain in the repo as siblings.

## Model architecture
- **Dual Graph Transformer layers** ([graphgps/layer/dgt_layer.py](../graphgps/layer/dgt_layer.py)): biased multi-head Q/K/V attention with pairwise features acting as **both** an attention bias ($E^{\text{att}}$) and a value modulator ($E^{\text{val}}$):

  $$ s_{i,j} \propto \exp\!\Big(\tfrac{Q_i K_j^\top}{\sqrt{d_k}} + E^{\text{att}}_{i,j}\Big), \quad h_i = \sum_j s_{i,j}(V_j + E^{\text{val}}_{i,j}) $$

  Layer pattern: `H = X + MLP(MHA(X, E_att, E_val)); X_out = BatchNorm(FFN(H))`. Variants: `DGTLayer`, `NodeGTLayer`, `EdgeGTLayer`.
- **Networks** ([graphgps/network/](../graphgps/network/)): `DGTModel`, `DGTModel3D`, `NodeGTModel`, `EdgeGTModel`, plus reference baselines (`gps_model`, `san_transformer`, `big_bird`, `custom_gnn`).
- **Dual-graph coupling**: the line-graph view (bond-as-node) is constructed via `line_graph` in [graphgps/transform/transforms.py](../graphgps/transform/transforms.py); atom and bond features are mutually fused into each other's attention via a `NodeEdgeEncoder` MLP.
- **Readout**: global average pooling over atom and bond representations, concatenated and passed through an MLP head (`line_graph` head in [graphgps/head/](../graphgps/head/)).

## Graph-specific encodings (pairwise feature matrix $E$)
Constructed once per molecule and reused across all DGT layers.
- **SPDE — Shortest-Path Distance Encoding**: pairwise embedding indexed by shortest-path length (1-hop, 2-hop, …); precomputed by `compute_shortest_paths` and used through `topology_edge_encoder.py`. Configurable cap via `dataset.spd_max_length` (default 8 in shipped configs).
- **RWPE / RWSE — Random-Walk Positional / Structural Encoding**: distribution of $k$-step random-walk visitation probabilities; implemented in `kernel_pos_encoder.py` and configured via `posenc_RWSE`. Configs use walks `range(1,17)` (16 steps).
- **RSE — Ring Structural Encoding**: ring sizes detected by `add_rings` are embedded as learnable vectors and added to pairwise features. Configurable via `dataset.rings_max_length` (default 18, except BACE/FreeSolv).
- Additional encoders supported: Laplacian PE, SignNet PE, EquivStable Laplacian PE, relative-PE edge encoder, type-dict encoder, OGB / PPA / AST / VOC-superpixel encoders.

## 3D geometric embedding (DGT(3D))
Implemented inside `PositionEncoder` in [graphgps/network/dgt_model.py](../graphgps/network/dgt_model.py):
- **Enveloped Bessel basis** (`BesselBasisLayer` + `Envelope`) for radial / interatomic-distance encoding — smooth decay to zero at the cutoff, with vanishing 1st and 2nd derivatives (envelope-polynomial trick).
- **Spherical harmonics** (`SphericalBasisLayer`) for bond–bond angles, ensuring rotational consistency.
- **Bond lengths** are injected into the bond feature matrix $N^b$; **atom–atom distances** into the atom-graph pairwise matrix $E^a$; **bond–bond angles** into the bond-graph pairwise matrix $E^b$.

## End-to-end data flow

> **Note on the DGT layer vs. the GPS layer.** The upstream `GPSLayer` ([graphgps/layer/gps_layer.py](../graphgps/layer/gps_layer.py)) is a *hybrid* "local MPNN ‖ global attention" block. **DGT does not use it.** `DGTLayer` ([graphgps/layer/dgt_layer.py](../graphgps/layer/dgt_layer.py)) is pure biased multi-head attention with no MPNN side branch, run in two parallel paths (atoms / bonds). The "MPNN-like" behaviour in DGT is provided by (a) the dense pairwise tensors $E^{\text{att}} / E^{\text{val}}$, which already encode one-hop and graph-structural information, and (b) the one-shot `NodeEdgeEncoder` cross-fusion that runs before the first attention layer.

Symbols used below: `B` = batch size, `N` = atoms in a molecule, `M` = undirected bonds, `D` = `cfg.gnn.dim_inner` (hidden width), `H` = `cfg.gt.n_heads`. `N_max` / `M_max` = max atom / bond counts across the molecules in the current mini-batch (dynamic per batch). PyG's `to_dense_batch` pads each molecule to `N_max` (or `M_max`) and emits a boolean mask so padded positions don't participate in attention. Memory for attention is therefore `O(B · N_max² · D)`, which is why batch sizes shrink for datasets with larger molecules (e.g. Lipophilicity).

### Stage 0 — Raw inputs (PyG `Data` per molecule)

```
x          [N, 9]       atom features (categorical, MoleculeNet encoding)
edge_index [2, 2M]      directed atom-graph edges
edge_attr  [2M, 3]      bond features (categorical)
pos        [N, 3]       3D atom positions          ← only if DGTModel3D
y          [1]          target label
```

### Stage 1 — Pre-transforms (one-shot, cached on disk)

Implemented in [graphgps/transform/transforms.py](../graphgps/transform/transforms.py):

```
add_rings              → rings        [N, ring_max+1]   RSE: ring-size membership
compute_shortest_paths → spd          [N, N]            SPDE: pairwise shortest-path length
line_graph             → e_batch      [M]               which molecule each bond belongs to
                         e2e_edge_idx [2, |E^b|]        bond-bond adjacency (two bonds share an atom)
(optional RWSE precompute)
                       → pestat_RWSE  [N, K]            random-walk return probabilities (K=16 in BBBP)
```

### Stage 2 — Feature encoder (`FeatureEncoder` in [graphgps/network/dgt_model.py](../graphgps/network/dgt_model.py))

Five sub-modules run in sequence per `forward`:

```
LinearNodeEncoder
   x [N, 9]                                ──▶ x [N, D]               # atom vectors
LinearEdge+RWSE-SPDE  (composed_edge_encoders.py)
   edge_attr [2M, 3] + spd[i,j] + RWSE     ──▶ edge_attr [2M, D]      # per-directed-edge
   plus dense pairwise tensors built:
       edge_attention,
       edge_values        [B, N_max, N_max, D]   # atom-atom pairwise → E^a_att / E^a_val
       e2e_edge_dense,
       e2e_edge_attention,
       e2e_edge_values    [B, M_max, M_max, D]   # bond-bond pairwise → E^b_att / E^b_val
NodeEdgeEncoder    ← atom ↔ bond cross-fusion AND the one MPNN step in DGT
   bond vector  e[k]  = MLP([ edge_attr[k] , x[src(k)] + x[dst(k)] ])         # [M, D]
   atom vector  x[i]  = MLP([ x[i] , scatter_sum(edge_attr, edge_index[1]) ]) # [N, D]
   #                              └─── this is one round of message passing:
   #                                   sum-aggregate edge features into each
   #                                   destination atom, then update with MLP.
   # DGT performs this MPNN step EXACTLY ONCE, before the transformer layers —
   # in contrast to the upstream GPSLayer which runs an MPNN at every layer.
PositionEncoder   ← only in DGTModel3D
   atom–atom distance → Linear      → added to edge_attention / edge_values    (E^a)
   bond–bond angle    → spherical-harmonics → Linear → added to e2e_edge_*     (E^b)
   bond length        → Linear      → bdl_ebd  (added to bond features below)
Dense-batch packing
   x [N, D]  ──to_dense_batch──▶  x_dense [B, N_max, D]  + mask  [B, N_max]
   e [M, D]  ──to_dense_batch──▶  e_dense [B, M_max, D]  + e_mask
```

After Stage 2, the batch object carries:

| Tensor | Shape | Meaning |
|---|---|---|
| `batch.x` | `[N, D]` | atom vectors $N^a$ |
| `batch.e` | `[M, D]` | bond vectors $N^b$ |
| `batch.edge_attention` | `[B, N_max, N_max, D]` | $E^a_{\text{att}}$ atom–atom attention bias |
| `batch.edge_values` | `[B, N_max, N_max, D]` | $E^a_{\text{val}}$ atom–atom message modulator |
| `batch.e2e_edge_attention` | `[B, M_max, M_max, D]` | $E^b_{\text{att}}$ bond–bond attention bias |
| `batch.e2e_edge_values` | `[B, M_max, M_max, D]` | $E^b_{\text{val}}$ bond–bond message modulator |

### Stage 3 — Stacked `DGTLayer × L` ([graphgps/layer/dgt_layer.py](../graphgps/layer/dgt_layer.py))

Each layer runs **two independent attention paths** in parallel. Pairwise tensors $E^{a/b}_{\text{att/val}}$ are **fixed across all layers** — only `x` and `e` get updated.

```
─────────── Atom path — full shape trace for one block ───────────
   x [N, D]
     │  to_dense_batch
     ▼
   x_dense          [B, N_max, D]
   edge_attention   [B, N_max, N_max, D]    ← E^a_att, fixed across layers
   edge_values      [B, N_max, N_max, D]    ← E^a_val, fixed across layers
   attn_mask        [B, N_max, N_max]
     │
     │  Q_h = Linear(x_dense)               [B, N_max, D]
     │  K_h = Linear(x_dense)               [B, N_max, D]
     │  V_h = Linear(x_dense)               [B, N_max, D]
     │
     │  Q_h.view(B, N_max, H, D/H)          [B, N_max, H, D/H]    # multi-head reshape
     │  K_h.view(B, N_max, H, D/H)          [B, N_max, H, D/H]    # (V_h stays as [B, N_max, D])
     │  K_h *= 1 / √(D/H)
     │
     │  scores = einsum('bihk,bjhk->bijh', Q_h, K_h)
     │                                       [B, N_max, N_max, H]      # scalar per (i,j,h)
     │  scores.unsqueeze(-1)                 [B, N_max, N_max, H, 1]
     │
     │  ── padding mask ──
     │  attn_mask.view(B, N_max, N_max, 1, 1)
     │  scores -= 1e24 · (~attn_mask)        [B, N_max, N_max, H, 1]
     │
     │  ── E_att broadcast-add ──
     │  E_att.view(B, N_max, N_max, H, D/H)  [B, N_max, N_max, H, D/H]
     │  # PyTorch broadcasting on the last dim:
     │  #     scores [B, N_max, N_max, H, 1  ]  ┐
     │  #   + E_att  [B, N_max, N_max, H, D/H]  ┘  → [B, N_max, N_max, H, D/H]
     │  # the size-1 last dim of `scores` is virtually expanded to D/H,
     │  # so the scalar Q·K / √(D/H) is added to every per-feature value
     │  # of E_att at the same (i, j, h).
     │  scores = scores + E_att              [B, N_max, N_max, H, D/H]
     │  scores.reshape(...)                  [B, N_max, N_max, D]      # flatten H·D/H
     │
     │  scores = softmax(scores, dim=2)      [B, N_max, N_max, D]      # softmax over j
     │  scores *= dropout(attn_mask)         [B, N_max, N_max, D]
     │
     │  ── value aggregation + E_val ──
     │  h = einsum('bijk,bjk->bik', scores, V_h)   [B, N_max, D]
     │  h += (scores * E_val).sum(dim=2)           [B, N_max, D]
     ▼
   h_n [B, N_max, D] → unpad [N, D] → Linear → dropout → +residual → BatchNorm
     ▼
   FFN: Linear(D→2D) → activation → Linear(2D→D) → dropout → +residual → BatchNorm
     ▼
   batch.x ← new atom vectors

─────────── Bond path ─────────────────────────────────────────────
   identical block, but on (e, e2e_edge_attention, e2e_edge_values, e_attn_mask)
   with N_max replaced by M_max throughout
     ▼
   batch.e ← new bond vectors
```

Two non-obvious design choices worth noting:

- **The pairwise bias creates per-feature attention.** After `scores + E_att` is flattened, the score tensor has shape `[B, N_max, N_max, D]` — i.e. each (i, j) pair has **D distinct attention weights** (one per feature dim), not a single scalar per (i, j, head) like vanilla multi-head attention. Softmax is taken *per feature dim* over the emitter axis. This is a substantively richer attention than vanilla MHA — the pairwise tensors $E^{\text{att}}$, $E^{\text{val}}$ carry signal at the feature granularity.
- **V is single-head, Q and K are multi-head.** Only `Q_h` and `K_h` are reshaped to `[B, N_max, H, D/H]`; `V_h` stays `[B, N_max, D]`. The multi-head structure appears in the dot-product score and is used to broadcast `E_att` per head, then is collapsed back before value aggregation.

Within a layer the atom path can't see the bond path and vice versa. The two are coupled **only** at Stage 2 (NodeEdgeEncoder) and Stage 4 (readout). The pairwise tensors $E^{a/b}_{\text{att/val}}$ themselves are **never updated** by the layers — they are built once in Stage 2 and reused $L$ times.

### Stage 4 — Readout (`line_graph` head in [graphgps/head/](../graphgps/head/))

```
batch.x [N, D] ──GAP per molecule──▶ X^a_pool [B, D]
batch.e [M, D] ──GAP per molecule──▶ X^b_pool [B, D]

   concat                          [B, 2D]
     │
     │  MLP: Linear(2D→D) → ReLU → … → Linear(D→C)
     ▼
   logits [B, C]              (C = 1 for binary classif., C = #targets otherwise)
     │
     ▼
   loss(logits, y)   ← cross-entropy / MSE / L1 / focal / weighted variants
```

### One-glance summary

```
SMILES ── RDKit ──▶ x, edge_index, edge_attr, pos
                         │
            pre_transform│  rings, SPD, line-graph, RWSE
                         ▼
            FeatureEncoder  ┌──────────────────────┐
            (encodes feats, │  atom vectors  N^a   │
            builds E^a/E^b, │  bond vectors  N^b   │   ← cross-fused by NodeEdgeEncoder
            handles 3D)     │  E^a_att, E^a_val   │   ← from SPDE + RWSE + 3D dist
                            │  E^b_att, E^b_val   │   ← from line-graph adj + RSE + 3D angle
                            └──────────────────────┘
                         │
                         │  × L layers
                         ▼
            DGTLayer ┌──────────────────────────────┐
                     │  atom attn:  softmax(QK+E^a) │  pairwise E^a/E^b tensors
                     │  bond attn:  softmax(QK+E^b) │  fixed across layers
                     │  +FFN, +BatchNorm, +residual │
                     └──────────────────────────────┘
                         │
                         ▼
            Readout: MLP( GAP(N^a) ⊕ GAP(N^b) )  ──▶  logits
```

## Datasets, splits, and loaders
- Loader entry point: [graphgps/loader/master_loader.py](../graphgps/loader/master_loader.py), with dataset-specific modules in [graphgps/loader/dataset/](../graphgps/loader/dataset/) (`aqsol_molecules`, `chiral3d_molecule_net`, `voc_superpixels`, `coco_superpixels`, `malnet_tiny`).
- **MoleculeNet** datasets used by the paper: BBBP, Tox21, ClinTox, SIDER, BACE, HIV, ESOL, FreeSolv, Lipophilicity (PyG `MoleculeNet` interface).
- **QM9** with explicit scaffold splits provided under [datasets/QM9_split/](../datasets/QM9_split/) (SMILES + split files to be copied into `datasets/QM9/raw`).
- **Chiral3D MoleculeNet** for chirality-sensitive 3D tasks.
- **Splitting**: scaffold split, 80/10/10, implemented after the chemprop scaffold-split reference; selected per-config via `dataset.split_mode: scaffold` and managed by [graphgps/loader/split_generator.py](../graphgps/loader/split_generator.py).

## Training, optimization, losses
- **Optimizer**: AdamW with weight decay (typical 1e-2), gradient clipping enabled in configs.
- **Schedulers** ([graphgps/optimizer/extra_optimizers.py](../graphgps/optimizer/extra_optimizers.py)): cosine-with-warmup (default), reduce-on-plateau, step decay. Pretraining uses linear warmup (5 epochs) + cosine annealing as described in the paper.
- **Losses** ([graphgps/loss/](../graphgps/loss/)): MSE, L1, cross-entropy, weighted cross-entropy, focal loss, multilabel classification (plain / weighted / focal), subtoken prediction.
- **Heads** ([graphgps/head/](../graphgps/head/)): `san_graph`, `inductive_node`, `inductive_edge`, `ogb_code_graph`, plus the `line_graph` pooling head used by DGT.
- **Custom training loop**: [graphgps/train/custom_train.py](../graphgps/train/custom_train.py) (selected via `train.mode: custom`).
- **Pretraining / fine-tuning hooks**: [graphgps/finetuning.py](../graphgps/finetuning.py) (`load_pretrained_model_cfg`, `init_model_from_pretrained`) — used for DGT(3D) PCQM4Mv2 → QM9 transfer.

## Per-dataset DGT hyperparameters (Supplementary Table 4)
| Dataset | Batch | Layers | Heads | Hidden | LR | RW steps | Max SPD | Max ring |
|---|---|---|---|---|---|---|---|---|
| BBBP | 32 | 6 | 32 | 128 | 4e-4 | 16 | 8 | 18 |
| Tox21 | 64 | 5 | 16 | 128 | 1e-4 | 16 | 8 | 18 |
| ClinTox | 64 | 5 | 8 | 64 | 5e-5 | 16 | 8 | 18 |
| SIDER | 16 | 5 | 8 | 64 | 1e-4 | 16 | 8 | 18 |
| BACE | 128 | 5 | 8 | 64 | 2e-5 | 16 | 8 | 12 |
| HIV | 32 | 5 | 8 | 64 | 2e-5 | — | 8 | 18 |
| ESOL | 128 | 6 | 16 | 64 | 5e-5 | 16 | 8 | 18 |
| FreeSolv | 64 | 3 | 16 | 128 | 5e-5 | 6 | — | 6 |
| Lipophilicity | 24 | 5 | 48 | 384 | 5e-5 | 16 | 8 | 18 |
| QM9 | 512 | 10 | 16 | 128 | 2e-4 | 16 | 8 | 18 |

DGT(3D) inherits the QM9 configuration.

## Experiment tracking
- **Weights & Biases** ([graphgps/config/wandb_config.py](../graphgps/config/wandb_config.py)) and **TensorBoardX** for logging; per-experiment YAML toggles `wandb.use`.

## Interpretation
The paper interprets predictions with **Grad-SAM** (gradient × self-attention map): node importance is the Hadamard product of element-wise gradients of the target w.r.t. attention activations, averaged over heads and nodes — useful for downstream descriptor-attribution analysis.

## Configs
YAML configs under [configs/](../configs/), grouped by domain (`biophysics/`, `physical_chemistry/`, `physiology/`, `quantum_mechanics/`), declare model type (`DGTModel` / `DGTModel3D`), encoder stack (e.g. `LinearEdge+RWSE-SPDE`), GT depth / heads / hidden dim, ring & SPD caps, and the training schedule.
