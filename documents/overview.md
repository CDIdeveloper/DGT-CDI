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
- **Testing:** a [tests/](../tests/) suite in two tiers — fast unit tests covering the new loader / head / config (run by default: `pytest`), and one slow end-to-end regression test marked `e2e` (run explicitly: `pytest -m e2e`). The `e2e` test is re-run at each phase boundary to catch regressions in the core DGT pipeline. See per-phase Verify steps.

The plan is phased; each phase has a verification check, matching the "Goal-Driven Execution" guideline in [CLAUDE.md](../CLAUDE.md).

## Phase 0 — Environment & sanity check
- [X] Build the env from [environment.yaml](../environment.yaml) (`mamba env create -f environment.yaml`, `conda activate dgt`), then verify imports and CUDA:
  `python -c "import torch, torch_geometric, torch_scatter, graph_tool, rdkit; print(torch.cuda.is_available())"` must print `True`.
- [X] The device hardcode at [main.py:156](../main.py#L156) (`cfg.device = 'cuda:0'`) is correct on a CUDA box — no patch needed.
- [X] **Apply the `LD_PRELOAD` activation script** — fixes the libgomp clash between PyTorch's pip wheel (old bundled `libgomp`) and conda-forge's `graph-tool` (needs `GOMP_5.0`). Without it, the BBBP run crashes at the `rings` pre-transform when `get_rings()` does `import graph_tool` ([transforms.py:186](../graphgps/transform/transforms.py#L186)) with `GOMP_5.0 not found`. Run once on the remote with the `dgt` env active — it installs a conda activate-hook so the correct `libgomp` is preloaded automatically on every `conda activate dgt` (covers `main.py`, `pytest`, and any subprocess):
  ```bash
  mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
  echo 'export LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}' > $CONDA_PREFIX/etc/conda/activate.d/zz-libgomp.sh
  echo 'unset LD_PRELOAD' > $CONDA_PREFIX/etc/conda/deactivate.d/zz-libgomp.sh
  ```
  Then `mamba deactivate && mamba activate dgt` to load the hook; verify with `echo $LD_PRELOAD` (should print `$CONDA_PREFIX/lib/libgomp.so.1`). The single quotes in the `echo`s are deliberate — they keep `$CONDA_PREFIX` literal in the script files so it expands at *activation* time, not when you run the `echo`.
- [X] **Smoke test** — confirm the pipeline runs without crashing (BBBP auto-downloads via PyG; the first run also caches the `rings` / `SPD` / `line_graph` / `RWSE` pre-transforms):
  `python main.py --cfg configs/physiology/BBBP-RWSE-SPDE-Rings.yaml --repeat 1 seed 0 wandb.use False optim.max_epoch 3`
- [X] **Full reproduction with the DGT pipeline** — runs the canonical routine: train + val each epoch, test held out and run **once** on the best-val checkpoint, per-sample test predictions dumped for post-hoc analysis. Uses [BBBP-DGT-Pipeline.yaml](../configs/physiology/BBBP-DGT-Pipeline.yaml) (parallel alternative to the upstream config) and [graphgps/train/dgt_train.py](../graphgps/train/dgt_train.py) (`train.mode: dgt`).
  ```bash
  python main.py \
    --cfg configs/physiology/BBBP-DGT-Pipeline.yaml \
    --repeat 4 seed 0 wandb.use False
  ```
  Per-seed output lands in `results/DGT/BBBP-DGT-Pipeline/<seed>/`:
  - `train/stats.json`, `val/stats.json` — one JSON line per epoch.
  - `test/stats.json` — **one line only**, at the best-val epoch.
  - `test/predictions.pt` — per-sample `(y_true, y_pred)` for analysis.
  - `ckpt/<best_epoch>.ckpt` — single best-val checkpoint (`ckpt_best: True`, `ckpt_clean: True` enforced by the dgt train mode).
  - `agg/` — `agg_runs()` aggregates mean ± std across seeds at the end of `main.py`.
- [X] **Post-hoc analysis** — turn `predictions.pt` into plots (ROC, PR, confusion matrix @ optimal-F1, score histogram) using [scripts/analyze_run.py](../scripts/analyze_run.py). Run once per seed:
  ```bash
  for s in 0 1 2 3; do
    python scripts/analyze_run.py results/DGT/BBBP-DGT-Pipeline/$s
  done
  ```
  Outputs land under `<run_dir>/plots/` with a `summary.json` for the scalar metrics + best epoch.
- [X] **Record the chosen model** in [trained_models.md](trained_models.md) — date, config path, git SHA, chosen seed, best-val + test metrics, checkpoint path. See [modeling_routine.md](modeling_routine.md) for the full step-by-step.
- [X] code to train final model using the best parameter. train with/without test data.
- [X] generate code to make predictions for new data — see [scripts/predict.py](../scripts/predict.py) (binary classification + single-target regression; cuda-only). Deployment bundle = `<run_dir>/final_model{,_with_test}.{ckpt,config.yaml,json}` (the three sibling files are written by `retrain_on_trainval.py`; copy the trio together to any other server). Worked example in [modeling_routine.md → Step 7](modeling_routine.md#step-7--predict-on-new-data).
- [X] re-run training of final model see if all 3 files are saved together: three `final_model{,_with_test}.*` files. its under results/DGT/<data_name>-Pipeline/
- [X] test on a regression task. added FreeSolv-DGT-Pipeline.yaml, command:
  `
  python main.py \
  --cfg configs/physical_chemistry/FreeSolv-DGT-Pipeline.yaml \
  --repeat 2 seed 0 wandb.use False optim.max_epoch 50
  `
  and train final model for this regression case:
  `
  python scripts/retrain_on_trainval.py results/DGT/FreeSolv-DGT-Pipeline/ --include-test
  `
- [X] **Follow-up to `predict.py`:** extend to regression task type (output schema: just `y_pred`, no threshold / `y_pred_label`). `--threshold` is silently ignored when `cfg.dataset.task_type == 'regression'`.
- [X] test predict.py, for classification:
  `
  python scripts/predict.py \
  --ckpt results/DGT/BBBP-DGT-Pipeline/final_model.ckpt \
  --smiles-csv tests/sample_smiles.csv \
  --output-csv /tmp/predict_out.csv
  `
  and for regression (requires the FreeSolv-DGT-Pipeline run above to have finished and `retrain_on_trainval.py` to have been run):
  `
  python scripts/predict.py \
  --ckpt results/DGT/FreeSolv-DGT-Pipeline/final_model_with_test.ckpt \
  --smiles-csv tests/sample_smiles.csv \
  --output-csv /tmp/predict_out_reg.csv
  `

## Phase 1 — Dataset integration (biodegradability)

> **Data flow.** S3 stays canonical. A one-shot prepare script fetches a local snapshot into `datasets/<dataset_name>/raw/`. PyG's `processed/` cache handles all subsequent training runs without re-touching S3 or AWS credentials.
>
> **Scope.** Phase 1 implements the integration for the `biodeg_gwu` dataset (the GWU batch-2 variant from [trans_learn DATASET_REGISTRY](../tests/data_loading/settings.py)) — first because it's the primary biodegradability dataset we want to model. Once the pipeline works end-to-end on `biodeg_gwu`, swapping to the no-Reaxys `biodeg` dataset is just a `--dataset biodeg` re-run of the prepare script + a new YAML (the loader and model code stay the same). Other registered datasets (`abs`, `emi`, `qy`, `extin`) are deferred to [Future work](#future-work-post-phase-6) because they need solvent handling.

- [X] Write `scripts/prepare_data.py` — generic preparation script driven by `--dataset <name>` where `<name>` matches a key in trans_learn's `DATASET_REGISTRY`. Phase 1 runs it for `biodeg_gwu`; `biodeg` (and later, other datasets) re-use the same script unchanged. The script:
  - [X] Imports `DatasetLoader` (from [tests/data_loading/load_data.py](../tests/data_loading/load_data.py)) and trans_learn's `DATASET_REGISTRY` / `S3Handler`. Trans_learn path is passed via `--trans-learn-path`; auto-detects whether the path is the repo root or the package dir itself.
  - [X] Calls `DatasetLoader.load_split('train')` and `load_split('test')` with the existing preprocessing (NaN-fill, drop-columns, target-NaN filter; normalisation **off** — descriptor normalisation is Phase 2's job to keep ablation-clean).
  - [X] Writes the result locally to `datasets/<name>/raw/{train,test}.parquet`. Each parquet keeps the trans_learn layout: first `id_column_count` columns hold SMILES + identifiers + target `y`; remaining columns are descriptors.
  - [X] Also writes `datasets/<name>/raw/manifest.json` carrying `{id_column_count, id_columns, target_column, smiles_column, descriptor_columns, desc_dim, n_train, n_test, task_type_hint}` — so the PyG loader doesn't need to import trans_learn at runtime.
  - [X] Idempotent — re-running overwrites the local snapshot from S3.
- [X] Run the prepare script for `biodeg_gwu` on the remote and confirm the three output files appear under `datasets/biodeg_gwu/raw/`. **Result:** 5742 train + 300 test rows, 247 descriptors, binary classification target. `.gitignore` policy: **exclude** `datasets/biodeg_gwu/raw/` (the existing `.gitignore` rule covers it).
- [X] Implement `graphgps/loader/dataset/biodeg_gwu.py` as a PyG `InMemoryDataset` (pattern: [chiral3d_molecule_net.py](../graphgps/loader/dataset/chiral3d_molecule_net.py) / [aqsol_molecules.py](../graphgps/loader/dataset/aqsol_molecules.py)). Reads `datasets/biodeg_gwu/raw/{train,test}.parquet` + `manifest.json`; caches featurised graphs in `datasets/biodeg_gwu/processed/`. **No trans_learn import** — the loader stays self-contained.
  - [X] Per molecule: SMILES → RDKit `Mol` → atom/bond features (reuse OGB / MoleculeNet featurisation), `data.y` ∈ {0, 1}, `data.desc` = descriptor vector stored as shape `[1, desc_dim]` (graph-level attribute, so PyG collation stacks it to `[B, desc_dim]`). **No normalisation here** — descriptors are stored as-loaded; Phase 2 standardises them.
  - [X] Split each parquet's columns into SMILES/identifiers/y vs descriptors using `id_column_count` from the manifest.
  - [X] Set `train_graph_index` / `val_graph_index` / `test_graph_index` on `dataset.data` — lists of indices, the graph-level split convention (`*_mask` is the node-level convention and does not apply here). Train/test split is inherited from the prepare step (which parquet each row came from); ~10 % of train is carved out as validation with a fixed seed for reproducibility.
  - [X] Drop unparseable SMILES with a logged warning (don't silently skip). **Result:** zero rows dropped on first run.
- [X] Register a `PyG-biodeg_gwu` format in [graphgps/loader/master_loader.py](../graphgps/loader/master_loader.py). **No change to [graphgps/loader/split_generator.py](../graphgps/loader/split_generator.py) is needed** — the existing `split_mode: standard` already consumes pre-set `*_graph_index` attributes for graph-level tasks ([split_generator.py:68-74](../graphgps/loader/split_generator.py#L68)).
- [X] **Dataset-level smoke test** — load via `BiodegGwu('datasets/biodeg_gwu')` and inspect splits + class balance. **Result:** train n=5168 (43.1% pos), val n=574 (41.8% pos), test n=300 (48.0% pos), desc_dim=247.
- [X] **End-to-end smoke test** — baseline config [configs/biodegradability/Biodeg-GWU-DGT-Pipeline.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline.yaml) (mirrors `BBBP-DGT-Pipeline.yaml`; `gnn.head: line_graph` — no descriptor fusion yet, that's Phase 2). 3-epoch dry-run:
  ```bash
  python main.py \
    --cfg configs/biodegradability/Biodeg-GWU-DGT-Pipeline.yaml \
    --repeat 1 seed 0 wandb.use False optim.max_epoch 3
  ```
  **Result (2026-05-28):** exit code 0; converged (train_loss 0.72 → 0.54, val AUC 0.43 → 0.81); test AUC = **0.8115** at best-val epoch 2; 300 test rows tested automatically by `dgt_train`. Cleared three pre-existing fork bugs along the way (see "Fork bugs cleared during Phase 1" below).
- [ ] **After `biodeg_gwu` works end-to-end:** repeat the same steps for the `biodeg` (no-Reaxys) dataset.
  - [X] Code in place: [graphgps/loader/dataset/biodeg.py](../graphgps/loader/dataset/biodeg.py) (sibling of biodeg_gwu loader; same featurisation + split convention); `preformat_Biodeg` registered as `PyG-biodeg` in [master_loader.py](../graphgps/loader/master_loader.py); `PyG-biodeg` branch added to [linear_edge_encoder.py](../graphgps/encoder/linear_edge_encoder.py); baseline config at [configs/biodegradability/Biodeg-DGT-Pipeline.yaml](../configs/biodegradability/Biodeg-DGT-Pipeline.yaml).
  - [ ] Run the prepare script: `python scripts/prepare_data.py --dataset biodeg --trans-learn-path /home/jovyan/tools/trans_learn`.
  - [ ] Inspect manifest + class balance via the dataset-level smoke test (same one-liner as for biodeg_gwu, swapping the path to `datasets/biodeg`).
  - [ ] Run 3-epoch dry-run end-to-end with `python main.py --cfg configs/biodegradability/Biodeg-DGT-Pipeline.yaml --repeat 1 seed 0 wandb.use False optim.max_epoch 3` to confirm the full pipeline works.
  - [ ] Full 4-seed run + record in [trained_models.md](trained_models.md) HPO sweeps table (or just baseline if you're not exploring HPO for this dataset yet).

> **Fork bugs cleared during Phase 1** (would have bitten any fresh dataset, not just biodeg_gwu — BBBP didn't trigger them because the user happened to have a previously-compatible graph_tool installed):
> 1. [graphgps/transform/transforms.py](../graphgps/transform/transforms.py) `get_rings()` — `graph_tool.stats` submodule not auto-imported on conda-forge builds, *and* `remove_self_loops` / `remove_parallel_edges` moved from `graph_tool.stats` to `graph_tool.generation` in graph_tool 2.45+. Fixed via try/except (handles both old + new builds).
> 2. [graphgps/encoder/linear_edge_encoder.py](../graphgps/encoder/linear_edge_encoder.py) `LinearEdgeEncoder.__init__` — hardcoded format dispatcher rejected `PyG-biodeg_gwu`. Fixed with a one-line elif; deeper refactor logged in [Future work](#future-work-post-phase-6).

## Phase 2 — Descriptor plumbing (late fusion)
- [ ] Standardise descriptors (z-score using train-set mean/std; persist stats so test/val use the same normalisation).
- [ ] Carry descriptors through the DGT backbone untouched — `batch.desc` is a graph-level tensor `[B, desc_dim]` produced directly by PyG's mini-batch collation. It does **not** pass through `to_dense_batch` (that only applies to node-level tensors), so no backbone code needs to change.
- [ ] Add `DescriptorGraphHead` under [graphgps/head/](../graphgps/head/) — same as the current `line_graph` head (`LineGraphHead`, [san_graph.py:57](../graphgps/head/san_graph.py#L57)) but concatenates `batch.desc` (optionally passed through a small MLP) before the final `out_layer`. Register via `register_head` (e.g. as `line_graph_with_desc`).
  - [ ] **Matching marker comment.** Add a `MOLECULAR DESCRIPTORS CONSUMED HERE` comment block at the line where `batch.desc` is read inside the head's `forward()`, paired with the entry-point markers already in [biodeg_gwu.py](../graphgps/loader/dataset/biodeg_gwu.py) and [biodeg.py](../graphgps/loader/dataset/biodeg.py) (`MOLECULAR DESCRIPTORS ENTER HERE`). Together they form a closed set — grepping for `MOLECULAR DESCRIPTORS` should return exactly: **one ENTER per dataset that carries descriptors** (currently 2: biodeg_gwu + biodeg) **+ one CONSUMED per head variant that uses them** (currently 0; Phase 2 adds 1 via `line_graph_with_desc`). Any other match means someone routed `desc` through the backbone — investigate immediately.
- [ ] Add `tests/test_descriptor_head.py` — instantiate `DescriptorGraphHead`, run a forward pass on a toy batch, assert output shape `[B, C]`; assert both `line_graph` and `line_graph_with_desc` instantiate.
- [ ] **Verify:** `pytest tests/test_descriptor_head.py` passes; the ablation switch (`gnn.head: line_graph` vs `line_graph_with_desc`) toggles cleanly; `pytest -m e2e` still green.

## Phase 3 — Config & first training run

> **Baseline already exists.** [configs/biodegradability/Biodeg-GWU-DGT-Pipeline.yaml](../configs/biodegradability/Biodeg-GWU-DGT-Pipeline.yaml) (created in Phase 1's end-to-end smoke test) is the *no-descriptor* baseline. Phase 3 adds the *with-descriptor* variant alongside it.

- [ ] Register the new `dataset.desc_dim` config field in [graphgps/config/dataset_config.py](../graphgps/config/dataset_config.py) — GraphGym rejects unknown YAML keys, so the field must exist in the schema before a config can reference it.
- [ ] Create `configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc.yaml`, a copy of the Phase 1 baseline with two changes:
  - [ ] `gnn.head: line_graph_with_desc` (was: `line_graph`).
  - [ ] `dataset.desc_dim: 247` (the value from `datasets/biodeg_gwu/raw/manifest.json`).
  - [ ] Loss: keep `cross_entropy` — Phase 1 confirmed ~43% positive across splits, well-balanced. (If a later dataset comes in materially imbalanced (>~70/30), switch to `weighted_cross_entropy` or `focal_loss`, both already in [graphgps/loss/](../graphgps/loss/).)
- [ ] Add `tests/test_config.py` — assert the new `Biodeg-GWU-DGT-Pipeline-WithDesc.yaml` loads without error and that `dataset.desc_dim` is a recognised config key.
- [ ] Run `python main.py --cfg configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc.yaml --repeat 4 seed 0`.
- [ ] **Verify:** `pytest tests/test_config.py` passes; val ROC-AUC > random; test ROC-AUC is logged; no NaN losses; W&B run visible; `pytest -m e2e` still green.

## Phase 4 — Ablation: does the descriptor channel help?
- [ ] Train two DGT variants (4 seeds each, same data, same split, both with `train.mode: dgt`), toggled purely by the `gnn.head` config key:
  - [ ] DGT only (`gnn.head: line_graph`).
  - [ ] DGT + descriptors (`gnn.head: line_graph_with_desc`).
- [ ] Add a descriptors-only baseline — a small standalone MLP on the descriptor vector (a separate script / model, **not** a `gnn.head` toggle of `DGTModel`) — as a quick sanity check that the descriptors carry signal on their own.
- [ ] For every variant × seed, run [scripts/analyze_run.py](../scripts/analyze_run.py) to generate ROC / PR / confusion-matrix plots + a `summary.json` with the scalar metrics:
  ```bash
  for cfg in Biodeg-GWU-DGT-Pipeline Biodeg-GWU-DGT-Pipeline-WithDesc; do
    for s in 0 1 2 3; do
      python scripts/analyze_run.py results/DGT/$cfg/$s
    done
  done
  ```
- [ ] Aggregate the per-seed `summary.json` files into a comparison table (mean ± std of test ROC-AUC, AUPRC, accuracy at the optimal-F1 threshold) — this is the ablation report.
- [ ] Record the winning model in [trained_models.md](trained_models.md).
- [ ] **Verify:** clear ordering and confidence intervals; decide whether to keep descriptors in the default config.

## Phase 5 — Interpretation (optional, paper-aligned)
- [ ] Implement Grad-SAM-style attention attribution (Supplementary Information §9 "Attention-based interpretation", in `documents/paper/supp_nc_paper_lean.docx`) over the DGT attention maps.
- [ ] Pick a handful of RD vs NRD molecules, render atom-level importance overlays, sanity-check against known biodegradability substructures (e.g., ester / amide hydrolysis sites, halogenation patterns).
- [ ] **Verify:** importance maps are non-trivial (not all-uniform) and qualitatively reasonable.

## Phase 6 — Pretraining (optional, conditional)

**Gate — only undertake this if *both* hold after Phase 4:**
1. the biodegradability dataset is small (roughly < a few thousand molecules), and
2. the Phase 4 baseline shows a generalisation gap (over- or under-fitting) that more data could plausibly close.

Note: this is **not** the paper's pretraining recipe. The paper's `### Pretraining setup` ([nc_paper_lean.md](paper/nc_paper_lean.md)) is specific to **DGT(3D)** — supervised transfer of a *quantum* property (PCQM4Mv2 HOMO–LUMO gap → QM9 HOMO/LUMO), which has no mechanistic link to a 2D biodegradability classification task.

- [ ] Scope: **supervised transfer** with the 2D `DGTModel`. Pretrain on a larger MoleculeNet classification dataset (e.g. Tox21 ~8k, or HIV ~41k), then fine-tune on biodegradability.
- [ ] Use the existing weight-transfer machinery — [graphgps/finetuning.py](../graphgps/finetuning.py) (`load_pretrained_model_cfg`, `init_model_from_pretrained`) — driven by `cfg.pretrained.dir`, `cfg.pretrained.freeze_main`, `cfg.pretrained.reset_prediction_head`. The biodegradability head is reset; the DGT backbone is initialised from the pretrained weights.
- [ ] Atom/bond featurisation must match between the pretraining and fine-tuning datasets (same encoder, same `dim_in`) so the backbone weights are transferable.
- [ ] **Verify:** fine-tuned-from-pretrained test ROC-AUC is compared against the from-scratch Phase 4 result over 4 seeds; keep pretraining only if it shows a clear, consistent gain.
- Out of scope: self-supervised pretraining (masked-atom / contrastive) — not implemented in this repo, and a separate larger effort.

## Future work (post-Phase 6)

- [ ] **Prepare other `trans_learn` datasets for DGT training.** `scripts/prepare_data.py` from Phase 1 is generic; running it for the other dataset names in [DATASET_REGISTRY](../tests/data_loading/settings.py) lands their parquets under `datasets/<name>/raw/`. Datasets in scope (in roughly increasing complexity, after the biodegradability variants in Phase 1): `extin`, `abs`, `emi`, `qy`. All four are *regression* (continuous target) and need *solvent fusion* — see [config_reference.md](config_reference.md) for the YAML changes needed (`task_type`, `metric_best`, `metric_agg`, `loss_fun`) and the next bullet for solvent handling.
- [ ] **Solvent fusion — needed for `abs` / `emi` / `qy` / `extin`** (solvent is meaningful for these tasks; not for biodeg). **Default approach: solvent descriptor channel** — concatenate solvent-derived features (categorical solvent embedding or solvent descriptors from the same dataset parquet) into the same head channel as molecular descriptors. Minimal architectural change beyond Phase 2's `line_graph_with_desc` head; reuses the descriptor-fusion mechanism.
- [ ] **Alternative solvent fusion methods (escalate only if descriptor channel plateaus):**
  - **Two-tower:** run `DGTModel` twice — once on solute, once on solvent — concatenate both graph embeddings before the head. Roughly 2× compute; needs a new head that takes two pooled vectors.
  - **Joint graph:** pack solute + solvent into one `Data` object with an extra graph-id tensor; let attention mix the two graphs. Most expressive, most invasive (encoder + head both touched).
- [ ] **Solvent-channel cleanup if pursued:** decide whether the solvent SMILES column needs to be carried as a separate column to the loader (for option 2/3) or only its derived descriptors (option 1). Manifest schema in Phase 1's prepare script may need a `solvent_smiles_column` field.
- [ ] **Refactor [graphgps/encoder/linear_edge_encoder.py](../graphgps/encoder/linear_edge_encoder.py)** to drop the `if cfg.dataset.format == 'PyG-...'` chain in `LinearEdgeEncoder.__init__`. The error message in the existing code already hints at the fix: *"refactor to use a cfg option."* Just always use `cfg.dataset.edge_encoder_num_types` (which every molecular dataset config already sets) and special-case only the genuinely-different non-molecular cases (`MNIST` / `CIFAR10`, which use `in_dim=1`). Net effect: adding a new molecular dataset no longer requires editing this file. Touch point during Phase 1 of biodeg_gwu showed up as a one-line elif addition — workable, but fragile, and the same pattern likely exists in sibling encoders worth auditing together.
- [ ] **Automate hyperparameter exploration.** Round 1 of biodeg_gwu HPO is currently a manual loop: hand-edit a YAML, launch `main.py`, wait, run `analyze_run.py` per seed, copy numbers into [trained_models.md](trained_models.md). Workable but tedious; doesn't scale once we want to sweep more than 3 variants × 4 seeds at a time. Automation candidates, in increasing complexity:
  - **Sweep driver script** — a small `scripts/run_sweep.py` taking a list of YAMLs (or a base YAML + a list of override dicts) and looping the launch + analyze_run + result-extraction steps; writes the comparison table directly to `trained_models.md` (or a side file). One weekend's work. Best ROI for our scale.
  - **GraphGym `--repeat` sweep extension** — GraphGym natively supports `grid sweeps` via its `agg_runs` / `mark_done` machinery. Wiring that up gives full multi-variant sweeps in one command, but the harness conventions are GraphGym-flavoured and may need tweaks for our `dgt_train` mode.
  - **External HPO library** (Optuna / Ray Tune) — overkill for a single-developer cloud-only project, only worth it if the search space gets large (e.g. >20 simultaneous variants). Defer.
  - In all cases the deliverable is: one command that takes a sweep spec → produces a populated [trained_models.md → HPO sweeps](trained_models.md) table, ready for human read-off. Manual filling stays as the fallback for one-off variants.
- [ ] **K-fold cross-validation as an alternative to the fixed 10 % val carve-out** for biodeg_gwu (and similar separate-test-parquet setups). The current biodeg_gwu loader holds out 574 training rows (10 % of `train.parquet`) as a fixed validation set; that's 574 molecules the model never sees during training. K-fold CV uses every training sample in K-1 of the folds, giving both a more efficient use of the ~5700 training molecules AND a more robust val metric (mean ± std over K folds). **Note:** GraphGym's harness already supports `split_mode: cv-kfold-K` / `cv-stratifiedkfold-K` (see [split_generator.py:setup_cv_split](../graphgps/loader/split_generator.py#L222) and `cfg.run_multiple_splits` in [main.py:run_loop_settings()](../main.py#L100)), so K-fold CV is **not greenfield work** — but the existing implementation folds across the *entire* dataset, mixing biodeg_gwu's train.parquet AND test.parquet rows into the same fold pool, which violates our "test.parquet stays held out" convention. The actual task is therefore: (a) write a biodeg_gwu-aware CV variant that K-folds only the train.parquet rows and leaves test.parquet untouched for each fold's test eval; (b) train K models (one per fold); (c) ensemble the K test-set predictions (mean probability or majority vote). Per-seed × per-fold compute is K× the current cost; worth it only if the Phase-4 descriptor-fusion ablation lands near the noise floor (then averaging over folds is what gives the comparison enough resolution to call a winner). Apply to other datasets too once proven on biodeg_gwu.

## Open items / assumptions to confirm later
- [ ] Exact descriptor list and dimensionality (currently treated as a black-box vector of length `desc_dim`). Phase 1's manifest will fix `desc_dim` once the prepare script runs against the live `biodeg` data.
- [X] ~~Class balance of the RD/NRD labels — drives the loss choice in Phase 3.~~ **Resolved 2026-05-28 from biodeg_gwu Phase 1 smoke test: ~43% positive across all splits → plain `cross_entropy` is correct for Phase 3 (no weighted / focal loss needed).**

