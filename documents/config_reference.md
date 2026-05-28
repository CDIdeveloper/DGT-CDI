# DGT YAML config — field reference

Line-by-line reference for the DGT training configs. Built around two
worked examples that share the same model architecture but differ in
task type:

- [BBBP-DGT-Pipeline.yaml](../configs/physiology/BBBP-DGT-Pipeline.yaml) — **binary classification** (BBBP blood-brain-barrier permeability).
- [FreeSolv-DGT-Pipeline.yaml](../configs/physical_chemistry/FreeSolv-DGT-Pipeline.yaml) — **single-target regression** (FreeSolv hydration free energy).

For *which fields to tune* (with sweep ranges and impact tiers), see
[modeling_routine.md → Hyperparameter exploration](modeling_routine.md#hyperparameter-exploration).
For *per-dataset values from the paper*, see
[tech.md → Per-dataset DGT hyperparameters](tech.md#per-dataset-dgt-hyperparameters).

## At a glance — what differs between the two configs

| Concern | BBBP | FreeSolv |
|---|---|---|
| `task_type` | `classification_binary` | `regression` |
| `metric_best` / `metric_agg` | `auc` / `argmax` | `rmse` / `argmin` |
| `model.loss_fun` | `cross_entropy` | `mse` |
| `edge_encoder_name` | `LinearEdge+RWSE-SPDE` | `LinearEdge+RWSE` (no SPDE) |
| `dataset.spd` | `True` | omitted (False) |
| `dataset.rings_max_length` | `18` | `6` (smaller molecules) |
| `posenc_RWSE.kernel.times_func` | `range(1,17)` | `range(1,7)` |
| `train.batch_size` | `32` | `65` |
| `gt.attn_dropout` | `0.3` | `0.7` (stronger regularisation, tiny dataset) |
| `optim.weight_decay` | `1e-2` | `1e-5` |
| `optim.base_lr` | `4e-4` | `5e-5` |
| `optim.max_epoch` | `50` | `2000` |

Everything else (model class, layer count, hidden dim, head, pooling, encoders) is identical.

## Field-by-field reference (merged YAML)

Each line shows the BBBP value with the FreeSolv override in a comment when they differ. The comment explains what the field controls and how it affects training / performance. Where applicable, the **Tier** annotation refers to [modeling_routine.md → Hyperparameter exploration](modeling_routine.md#hyperparameter-exploration).

```yaml
out_dir: results/DGT
# Root output directory. Each config's outputs land under
# <out_dir>/<config_basename>/<seed>/. Rarely changed.

metric_best: auc                # FreeSolv: rmse
# Which metric drives best-val checkpoint selection — i.e. what counts as
# "the best epoch". Must be a key the logger writes to val/stats.json.
# Wrong choice → wrong checkpoint kept; downstream test eval / predict
# inherits the mistake.

metric_agg: argmax              # FreeSolv: argmin
# Direction for metric_best across epochs. `argmax` for metrics where
# higher is better (AUC, accuracy, F1, R²); `argmin` for lower-is-better
# (RMSE, MAE, loss). Mis-pairing with metric_best silently picks the
# worst epoch instead of the best.

wandb:
  use: False                    # Off for quick / offline runs.
  project: BBBP                 # FreeSolv: FreeSolv
# Weights & Biases tracking. Zero perf impact; pure logging.

dataset:
  format: PyG-MoleculeNet
  # Resolves to a loader factory in graphgps/loader/master_loader.py.
  # Other options: PyG-AQSOL, PyG-Chiral3DMoleculeNet, PyG-QM9, OGB-...
  # Determines which dataset class is built — not a tuning knob.

  name: BBBP                    # FreeSolv: FreeSolv
  # Sub-dataset within the chosen format. Other MoleculeNet names:
  # ESOL, Lipo, BACE, HIV, ClinTox, SIDER, Tox21, ...

  task: graph
  # Graph-level prediction (one label per molecule). Don't change for
  # molecular property prediction tasks.

  task_type: classification_binary    # FreeSolv: regression
  # Determines loss family, metric family, and head output dim. Other
  # values: classification_multilabel, regression. Mismatch with the
  # actual label format → unstable training or zero gradients.

  transductive: False
  # False for graph-level tasks (train/val/test are disjoint molecule
  # sets). Don't change.

  split_mode: scaffold
  # 80/10/10 split by Bemis-Murcko scaffold (chemprop-style). Tests
  # generalisation to unseen scaffolds — typically 5-10 AUC points
  # harder than random split, but a more realistic generalisation
  # estimate. Alternatives: `standard` (consume pre-set
  # *_graph_index attrs), `random`.

  node_encoder: True
  node_encoder_name: LinearNode
  # Project the 9-dim categorical atom features to D-dim via a small
  # learned embedding. Don't change unless you're changing the atom
  # featurisation (then dim_in changes too).
  node_encoder_bn: False
  # BatchNorm in the node encoder. Off — model has BatchNorm later.

  edge_encoder: True
  edge_encoder_shared: True
  # Share encoder weights between (i→j) and (j→i) directed edges.
  edge_encoder_name: LinearEdge+RWSE-SPDE    # FreeSolv: LinearEdge+RWSE
  # Composed encoder pipeline. Components joined with `+`; each must be
  # registered in graphgps/encoder/. FreeSolv omits SPDE because
  # `dataset.spd: False` — keep these two in sync or the encoder will
  # try to read attributes that the pre-transform didn't produce.
  edge_encoder_num_types: 3
  # Number of categorical bond feature types (bond_type, stereo,
  # is_conjugated). Fixed by MoleculeNet's featurisation.
  edge_encoder_bn: False

  spd: True                     # FreeSolv: commented out (False)
  spd_max_length: 8             # FreeSolv: not used
  # SPDE pre-transform — pairwise shortest-path distance as attention
  # bias. **Tier 3.** Captures global topology that pure attention can't
  # learn from short-range features alone. Helps most on datasets where
  # long-range pairs matter; less so on tiny molecules where SPD ≤ 3 for
  # most pairs. Changing spd_max_length **requires** cache invalidation:
  # `rm -rf datasets/<DatasetName>/processed/`.

  rings: True
  rings_max_length: 18          # FreeSolv: 6
  # RSE pre-transform — ring-size membership as attention bias. **Tier 3.**
  # Larger value covers macrocycles; smaller is sufficient for small
  # molecules (FreeSolv averages ~9 atoms). Larger = more embedding
  # params; rings beyond max are clipped. Change requires cache
  # invalidation.

  rings_coalesce_edges: False
  # Merge ring-adjacency entries with multiple edges. Off.

share:
  dim_in: 9
  # Input atom feature dim — fixed by MoleculeNet's 9-feature encoding.
  # Don't change unless changing the featurisation.

posenc_RWSE:
  enable: True
  enable_edges: True
  # Apply RWSE to edges (not just nodes). Required when
  # edge_encoder_name contains `RWSE`.
  kernel:
    times_func: range(1,17)     # FreeSolv: range(1,7)
  # Random-walk step lengths included in the structural encoding.
  # **Tier 3.** Each step adds a feature column. Longer walks capture
  # longer-range structure; on small molecules they rarely add
  # information (most walks return). Change requires cache invalidation.
  model: Linear
  dim_pe: 64
  # PE embedding dim, concatenated into edge features. Larger = more
  # PE capacity but pushes the rest of the features into proportionally
  # smaller weight.
  raw_norm_type: None
  # No normalisation of raw RW return probabilities pre-encoder.

train:
  mode: dgt
  # This fork's parallel alternative to upstream `custom`. Train + val
  # each epoch; test loader is held out and run ONCE on the best-val
  # checkpoint; per-sample test predictions dumped to
  # <run_dir>/test/predictions.pt. See `modeling_routine.md` Step 3.

  batch_size: 32                # FreeSolv: 65
  # **Tier 2.** Larger batches = smoother gradient, more GPU memory;
  # smaller = more update steps per epoch (higher exploration). Attention
  # memory scales as O(B · N_max² · D), so datasets with bigger molecules
  # (Lipo, HIV) must use smaller batches than BBBP / FreeSolv.

  eval_period: 1
  # Validate every N epochs. 1 = every epoch.

  enable_ckpt: True
  ckpt_best: True
  ckpt_clean: True
  # All three required for `train.mode: dgt`; the train mode enforces
  # them automatically if unset. Net effect: only the single best-val
  # checkpoint stays on disk. No perf impact, just storage hygiene.

model:
  type: DGTModel
  # Network class. DGTModel = 2D dual-graph transformer. Alternatives:
  # DGTModel3D (needs `pos` in the Data object), upstream GPSModel
  # (baseline). Resolved against network_dict at model build time.

  loss_fun: cross_entropy       # FreeSolv: mse
  # Loss function. Must match task_type. Classification options:
  # cross_entropy, weighted_cross_entropy, focal_loss. Regression:
  # mse, l1. Mismatch with task_type → silent failures (zero gradients
  # or NaN losses).

  edge_decoding: dot
  graph_pooling: add
  # Inherited GraphGym fields. The DGT `line_graph` head supplies its
  # own atom + bond pooling, so `graph_pooling` here is effectively
  # bypassed at readout — but GraphGym still consumes the field at
  # cfg validation, so leave it set.

gt:
  layer_type: None+DGT
  # Compound string: <local>+<global>. `None` = no local MPNN module;
  # each layer is pure attention over both atom and bond graphs. This
  # is what distinguishes DGT from upstream `GPSLayer` (which would
  # use e.g. `GINE+Transformer`). Don't change unless you're swapping
  # in a different DGT variant.

  layers: 4
  # **Tier 1.** Number of stacked DGT layers. More = more capacity, more
  # overfitting risk, slower. Paper sweeps 3-10 across datasets; the
  # paper's recommended values are in tech.md's per-dataset table.

  n_heads: 16
  # **Tier 2.** Number of attention heads. **Must divide dim_hidden.**
  # More heads = richer attention decomposition; too many = each head
  # gets too narrow a feature slice.

  dim_hidden: 128
  # **Tier 1.** Hidden width of the GT layers. Larger = more capacity;
  # memory scales as dim_hidden × N_max² for the pairwise tensors.
  # **Must equal gnn.dim_inner** — mismatch causes a shape error at
  # the head boundary.

  dropout: 0.0
  # **Tier 2.** FFN dropout. 0 here because BatchNorm regularises.
  # Increase to 0.1-0.2 if you see val overfitting.

  attn_dropout: 0.3             # FreeSolv: 0.7
  # **Tier 2.** Dropout on attention scores. FreeSolv uses 0.7 because
  # its dataset is tiny (~642 molecules) and prone to overfitting.
  # Sweep 0.1-0.4 for normal-sized datasets, higher for small ones.

  layer_norm: False
  batch_norm: True
  # DGT uses BatchNorm only.

gnn:
  head: line_graph
  # Readout head. `line_graph`: pool atom and bond reps separately,
  # concat, then MLP → logits. Phase-2 `line_graph_with_desc` will
  # additionally concat molecular descriptors before the MLP (planned).
  # **Note on naming:** the field lives under `gnn.*` for GraphGym
  # historical reasons (the framework groups all model-side config there
  # regardless of whether the architecture is actually an MPNN). It does
  # NOT mean the head participates in graph message passing — it's
  # purely the post-pooling readout MLP. Molecular descriptors enter
  # the model **only** here, never through the GT layers or encoders.

  layers_pre_mp: 0
  # No pre-attention MPNN. The single MPNN step in DGT happens inside
  # `NodeEdgeEncoder` cross-fusion (see tech.md → Stage 2), not as a
  # separate pre-layer.

  layers_post_mp: 3
  # **Tier 2.** Depth of the post-pooling MLP head. Sweep 2-3.

  dim_inner: 128
  # Hidden width of the head MLP. **Must equal gt.dim_hidden** —
  # mismatch causes a shape error at the readout boundary.

  batchnorm: True
  act: gelu
  dropout: 0.0
  agg: mean
  # Head MLP details. `agg` is consumed elsewhere (the `line_graph`
  # head supplies its own atom/bond mean-pool independently).

  normalize_adj: False
  # Adjacency normalisation flag — not used by DGT (no graph conv).

optim:
  clip_grad_norm: True
  # Gradient clipping for training stability. Always on.

  optimizer: adamW
  # AdamW (Adam + decoupled weight decay).

  weight_decay: 1e-2            # FreeSolv: 1e-5
  # **Tier 1.** L2 regularisation strength on the optimizer side. BBBP
  # uses strong decay (1e-2) because the scaffold split makes
  # generalisation hard; FreeSolv uses weak decay (1e-5) because the
  # dataset is small and the model needs to actually fit it. Sweep
  # 1e-1, 1e-2, 1e-3, 0.

  base_lr: 0.0004               # FreeSolv: 0.00005
  # **Tier 1 — most impactful single knob.** Cosine schedule decays
  # from this value to ~0. Too high → diverges; too low → stuck.
  # Per-dataset values from the paper are in tech.md's table. Typical
  # sweep: 1e-3, 4e-4, 1e-4.

  max_epoch: 50                 # FreeSolv: 2000
  # Total training epochs. Cosine schedule shape is parameterised on
  # this — changing it stretches/compresses the entire schedule.
  # FreeSolv uses 2000 because base_lr is ~8× smaller and the dataset
  # is ~5× smaller, so each epoch is fast and many are needed to
  # converge. With ckpt_best the script keeps the best-val ckpt
  # regardless, so over-training is "free" except for compute time.

  scheduler: cosine_with_warmup
  # Linear warmup → cosine decay to 0. Standard transformer schedule.

  num_warmup_epochs: 10
  # **Tier 2.** Linear-warmup duration before cosine kicks in.
  # Convention: ~10% of max_epoch. Too short = early-step instability;
  # too long = wasted budget at low LR.
```

## How to use this reference

1. **Reading a new config** — open the YAML side-by-side with this file; comments here explain what each field does.
2. **Tuning** — start from [modeling_routine.md → Hyperparameter exploration](modeling_routine.md#hyperparameter-exploration) for *which fields matter most* (Tier 1/2/3); come back here for the meaning of any field that's unfamiliar.
3. **Adapting to a new dataset** — copy the closest existing config (classification → start from BBBP, regression → start from FreeSolv), then change `dataset.name` / `task_type` / `loss_fun` / `metric_best` / `metric_agg` first, then the size-dependent knobs (`batch_size`, `rings_max_length`, RWSE `times_func`).
4. **Cache invalidation** — any change to a `dataset.*` field marked **Tier 3** above or to `posenc_RWSE.kernel.times_func` requires `rm -rf datasets/<DatasetName>/processed/` before re-running. See [modeling_routine.md → Cleanup by parameter category](modeling_routine.md#cleanup-by-parameter-category-quick-reference-for-hpo-sweeps).
