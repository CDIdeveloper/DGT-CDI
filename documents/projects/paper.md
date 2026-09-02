# DGT on biodegradability (`biodeg_gwu_no_ind`) — paper base

Working document for a publication on applying the Dual Graph Transformer (DGT) to
ready-biodegradability classification, with a molecular-descriptor late-fusion ablation.

**Status (2026-09-02):** configuration selected on validation with test suppressed; test set
then read **once** for the selected configuration (§5.3). Headline: DGT reaches
**F1 0.8610, ROC-AUC 0.9196** on the 278-molecule test set, exceeding both prior baselines.

Related: [gwu.md](gwu.md) (earlier study on the *different* `biodeg_gwu` dataset —
see §8 before citing it), [../dgt_porting_guide.md](../dgt_porting_guide.md) (protocol
this work follows), [../trained_models.md](../trained_models.md) (model registry).

---

## 1. Contribution

Prior work on this endpoint ([porting guide §3](../dgt_porting_guide.md)) established that
descriptor gradient boosting (HistGradientBoosting, CV-F1 ≈ 0.813) outperforms a chemprop
MPNN (CV-F1 ≈ 0.784), suggesting the endpoint is driven by composition and functional-group
content that a purely graph-based view under-represents. **The question this work asks is
whether a graph transformer with an explicit descriptor-fusion channel closes that gap**,
and — separately — whether the descriptor channel contributes at all once model selection is
made without reference to the test set.

The second question turns out to matter: a previously reported descriptor benefit on a
related dataset does not survive re-derivation on validation data (§8).

---

## 2. Dataset

`biodeg_gwu_no_ind`, GWU batch 2 with inherently-biodegradable (InD) rows removed.

| Property | Value |
|---|---|
| Task | Binary classification: 1 = readily biodegradable (RB), 0 = not readily (NRB) |
| Train | 5264 molecules (2466 RB / 2798 NRB; 46.9 % positive) |
| Test | 278 molecules (144 RB / 134 NRB; 51.8 % positive) |
| Validation | 526 molecules, carved from train (10 %, fixed seed 42) → 4738 train / 526 val |
| Descriptors | 247 total = 40 quantum-mechanical (`_gwu`) + 207 RDKit + functional-group |
| Target / SMILES columns | `degradable` / `smiles` |
| Split | **Fixed** train/test partition, inherited from the upstream release; never re-partitioned |

The train/test split is external and treated as immutable. A canonical-SMILES near-duplicate
audit on this split found **0 true cross-split duplicates**; the 51 fingerprint-identical
test↔train pairs are stereoisomers and aliphatic chain-length homologs, an artifact of
Morgan radius 2 ([porting guide §1](../dgt_porting_guide.md)). The split is therefore
retained as published, with no de-duplication applied.

Class balance is near-even in every split, so unweighted cross-entropy is used throughout;
no class weighting or focal loss was needed.

### 2.1 Descriptor feature sets

Upstream, identifier and metadata columns were renamed `*_gwu` → `*_fromgwu`, so that
**only genuine QM features carry the `_gwu` suffix**. This makes suffix-based selection
unambiguous. Column selection is by substring match; `'_gwu'` is not a substring of
`'_fromgwu'`, so the two sets are cleanly separated.

| Feature set | Selection rule | Columns |
|---|---|---|
| `none` | — (graph only) | 0 |
| `qm` | contains `_gwu` | 40 |
| `rdkit_fg` | excludes `_gwu` | 207 |
| `qm_rdkit` | all descriptors | 247 |

---

## 3. Model

Dual Graph Transformer: parallel biased multi-head attention over the atom graph and the
bond (line) graph, with fixed pairwise structural tensors shared across layers. The backbone
is unmodified from the reference DGT implementation.

| Component | Setting |
|---|---|
| Layers / heads / hidden dim | 4 / 16 / 128 |
| Attention dropout | 0.3 (feed-forward dropout 0) |
| Normalisation | batch norm |
| Node encoder | `LinearNode` (9 atom features) |
| Edge encoder | `LinearEdge + RWSE-SPDE` (3 bond features) |
| Positional / structural encodings | RWSE (dim 64, walk lengths 1–16); shortest-path distance ≤ 8; ring detection ≤ 18 |
| Readout | `line_graph` — atom and bond representations pooled separately, concatenated, MLP (3 post-MP layers) |
| Graph pooling | sum |

**Descriptor fusion.** For the descriptor arms the readout is replaced by
`line_graph_with_desc`: the standardised descriptor vector is projected by
`Linear(desc_dim → 128) → GELU` and concatenated with the pooled graph embedding
immediately before the output layer. The descriptor vector is a graph-level attribute that
rides mini-batch collation and **never enters the encoders, the attention layers, or the
pairwise tensors** — so the backbone is byte-identical across all four arms and the ablation
isolates the descriptor channel alone.

Parameter counts: 1,252,609 (graph-only) and 1,279,361 (`rdkit_fg`, observed). The head adds
`desc_dim × 128 + 256` parameters, giving 1,257,985 (`qm`) and 1,284,481 (`qm_rdkit`) by
derivation — the descriptor channel is ≤ 2.5 % of model capacity in every arm.

### 3.1 Training

AdamW, base LR 4 × 10⁻⁴, weight decay 10⁻², cosine schedule with 10 warm-up epochs,
gradient-norm clipping, batch size 32, 50 epochs, binary cross-entropy. Four seeds (0–3) per
configuration. Runtime ≈ 0.87 h per 4-seed configuration on a single GPU (≈ 13–16 s/epoch).

**Descriptor standardisation.** Descriptors are z-scored using mean and standard deviation
computed on the **training rows only, excluding the validation carve-out**. Constant columns
(σ < 10⁻⁸) are assigned σ = 1. The statistics and the ordered column names are persisted
alongside the processed dataset and re-used verbatim at validation, test, and inference time,
so the normalisation is identical everywhere and no distributional information leaks from
held-out data.

---

## 4. Experimental protocol

The protocol is the leak-free selection procedure specified in
[porting guide §2](../dgt_porting_guide.md), and is a substantive part of this work's claim.

1. **Model selection uses validation only.** Architecture and feature set were chosen from
   validation scores. No test number was consulted, displayed, or available to the decision
   at the time it was made.
2. **Selection metric.** F1 primary; ROC-AUC as tiebreak when F1 differences fall inside the
   seed standard deviation. The tiebreak was invoked here, and recorded before the test set
   was read (§6).
3. **Per-seed checkpointing.** Within each run, training and validation are evaluated every
   epoch; the checkpoint retained is the best-validation-AUC epoch. The test set is
   **never** touched during the epoch loop.
4. **Single test pass.** After training, the best-validation checkpoint is reloaded and the
   test set is scored exactly once, per seed, with per-sample predictions persisted for
   post-hoc analysis.
5. **Metrics from probabilities.** The model emits calibrated probabilities via sigmoid.
   ROC-AUC and average precision are computed from probabilities; accuracy, precision,
   recall, and F1 from the probability thresholded at **0.5** — matching the definitions
   used for the HGB and MPNN baselines so the numbers are directly comparable.
6. **Fixed split respected.** Configuration uses the pre-set split indices emitted by the
   loader; the random and scaffold split modes, which would re-partition the whole pool and
   readmit test molecules to training, are not used.

### 4.1 Leakage audit

The implementation was audited line-by-line against
[porting guide §7](../dgt_porting_guide.md). Outcome:

| # | Requirement | Status |
|---|---|---|
| 1 | Selection never sees test | **Pass** for this study (see §8 for the earlier study, which does not) |
| 2 | Test scored once | Pass per run; scored once per configuration per seed across the ablation (§9) |
| 3 | No fit-on-test | Pass for scalers — statistics are train-only and exclude the validation carve-out |
| 4 | Fresh model per fold | Pass — no warm-starting; each seed builds a new model |
| 5 | Inner validation ⊂ train | **Pass** — validation is carved from the train parquet; test rows are unconditionally tagged and never subdivided |
| 6 | External split respected | **Pass** |
| 7 | Probabilities, not hard labels | **Pass** |
| 8 | No de-duplication re-split | Pass — split used as published |

Two items were initially *not* satisfied by the deployment path. The first — the deployment
seed being chosen by median test score — was **fixed** (§9 item 3); seed and epoch budget are
now both validation-derived. The second remains: the deployment decision threshold is
optimised on test predictions (§9 item 4). It affects only
`predict.py --threshold optimal-f1`; every metric reported in this document uses the fixed
0.5 threshold and is unaffected.

---

## 5. Results — validation

Four seeds per configuration; mean ± population standard deviation across seeds. Each seed
contributes the metric at its own best-validation epoch.

### 5.1 Summary

| Feature set | desc_dim | Val F1 | Val ROC-AUC | Median best epoch (F1 / AUC) |
|---|---|---|---|---|
| `qm_rdkit` (all) | 247 | **0.8165 ± 0.0037** | 0.8853 ± 0.0026 | 32 / 32 |
| `rdkit_fg` (non-GWU) | 207 | 0.8164 ± 0.0065 | **0.8876 ± 0.0010** | 40 / 29 |
| `qm` (GWU only) | 40 | 0.8119 ± 0.0050 | 0.8829 ± 0.0015 | 37 / 32 |
| `none` (graph only) | — | 0.8115 ± 0.0062 | 0.8875 ± 0.0051 | 35 / 27 |

### 5.2 Per-seed values

| Feature set | Val F1 (seeds 0–3) | Val ROC-AUC (seeds 0–3) |
|---|---|---|
| `none` | 0.8053, 0.8061, 0.8144, 0.8202 | 0.8812, 0.8932, 0.8919, 0.8838 |
| `qm` | 0.8124, 0.8137, 0.8175, 0.8039 | 0.8808, 0.8845, 0.8843, 0.8822 |
| `rdkit_fg` | 0.8202, 0.8121, 0.8084, 0.8249 | 0.8892, 0.8866, 0.8872, 0.8874 |
| `qm_rdkit` | 0.8164, 0.8106, 0.8180, 0.8208 | 0.8865, 0.8877, 0.8810, 0.8859 |

### 5.3 Test — selected configuration only

`rdkit_fg` (non-GWU, 207 descriptors), read **once** after the selection in §6 was recorded.
4 seeds, mean ± population std, threshold 0.5. Source:
`results/DGT/BiodegNoInd-DGT-Pipeline-WithDesc-nongwu/agg/test/best.json`.

| Metric | DGT (this work) |
|---|---|
| Accuracy | 0.8552 ± 0.0047 |
| Precision | 0.8562 ± 0.0078 |
| Recall | 0.8663 ± 0.0199 |
| **F1** | **0.8610 ± 0.0066** |
| **ROC-AUC** | **0.9196 ± 0.0027** |

### 5.4 Comparison with prior models on the same 278-molecule test set

References from [porting guide §3](../dgt_porting_guide.md). All at threshold 0.5, all on the
identical fixed test split.

| Model | Feature set | F1 | ROC-AUC |
|---|---|---|---|
| **DGT (this work)** | `rdkit_fg` | **0.8610** | **0.9196** |
| HGB (gradient boosting) | `rdkit_fg` | 0.8500 | 0.9152 |
| HGB | `qm_rdkit` | — | 0.9185 |
| MPNN (chemprop) | `rdkit_fg` | 0.8522 | 0.8969 |
| MPNN | `qm` | 0.8462 | 0.8969 |
| MPNN | `qm_rdkit` | 0.8362 | 0.8911 |

**DGT ranks first on both metrics.** Read carefully, though:

- **vs MPNN — a clear win.** ROC-AUC +0.0227 over the best MPNN arm, roughly eight times
  DGT's own seed std. F1 +0.0088. The graph transformer is decisively the better graph model
  on this endpoint.
- **vs HGB — a win on F1, a tie on ROC-AUC.** F1 +0.0110 over HGB's best (≈1.7 seed std);
  ROC-AUC +0.0011 over HGB's best AUROC arm, which is well inside DGT's std of 0.0027 and
  should be reported as parity, not superiority.
- **Caveat on all comparisons.** The published HGB and MPNN figures are point estimates with
  no reported dispersion, and both were selected by 5-fold CV on train while DGT was selected
  on a single validation split (§9 item 1). With 278 test molecules, the sampling uncertainty
  on any single accuracy estimate is roughly ±0.04 at 95 % — larger than every gap in the
  table. Differences of ~1 point should be treated as suggestive.

**Answering §1's question:** a graph transformer *does* close the MPNN-to-HGB gap on this
endpoint — it matches descriptor gradient boosting on ranking quality and edges ahead on
thresholded F1. It does so, however, largely on graph structure: the descriptor channel that
was supposed to supply the missing composition signal contributes almost nothing (§7).

---

## 6. Selection decision

Applying the §2 rule, in order, before any test number was viewed:

1. **F1 primary.** `qm_rdkit` (0.8165 ± 0.0037) vs `rdkit_fg` (0.8164 ± 0.0065). The
   difference is 0.0001 — an order of magnitude inside either configuration's seed standard
   deviation. Declared a tie.
2. **Tiebreak on ROC-AUC.** `rdkit_fg` (0.8876) > `qm_rdkit` (0.8853).
3. **Selected: `rdkit_fg` (non-GWU, 207 descriptors).**

This decision, including the fact that the tiebreak was invoked, was recorded prior to
reading the test set, as [porting guide §2](../dgt_porting_guide.md) requires.

---

## 7. Observations

**The descriptor channel is approximately neutral on this dataset.** The best descriptor arm
improves validation F1 by 0.0050 over graph-only (0.8165 vs 0.8115) — roughly 0.8 standard
deviations — and validation ROC-AUC by 0.0001 (0.8876 vs 0.8875), which is indistinguishable
from zero. The full spread across all four arms is 0.0050 on F1 and 0.0047 on AUC, comparable
in magnitude to the seed-to-seed variation within a single arm.

**Quantum-mechanical descriptors do not contribute.** The `qm` arm (40 QM columns) ranks
third on F1 and last on ROC-AUC, and adding QM columns on top of RDKit (`qm_rdkit` vs
`rdkit_fg`) does not improve ROC-AUC (0.8853 vs 0.8876). This reproduces the qualitative
finding of the earlier study on `biodeg_gwu` (§8), and is consistent with an endpoint driven
by composition and functional-group content rather than electronic structure.

**Descriptors reduce ROC-AUC variance but not F1 variance.** The graph-only arm has the
largest ROC-AUC standard deviation (0.0051); `rdkit_fg` is five times tighter (0.0010). The
effect does not carry over to F1, where `rdkit_fg` has the *largest* spread (0.0065). This
asymmetry should not be over-interpreted from four seeds and is reported as an observation,
not a claim.

**Validation understates test performance here.** Test exceeds validation by a wide margin on
both metrics (F1 0.8610 vs 0.8164; ROC-AUC 0.9196 vs 0.8876). This is the opposite of the
usual direction and is not explained by optimism in the validation score, which is itself
maximised over epochs. The likely causes are the small test set (278 molecules) and its
composition: the test split is 51.8 % positive against 46.9 % in train. Validation was
therefore a conservative selection signal, which does not undermine the selection — the
ranking is what matters, not the level — but it means validation scores should not be quoted
as performance estimates.

**The two metrics disagree on the winner, and both top-two gaps are 0.0001.** ROC-AUC favours
`rdkit_fg`, F1 favours `qm_rdkit`. With four seeds and differences three orders of magnitude
below the metric values, the honest reading is that **the four configurations are not
distinguishable at this sample size**; the selection in §6 is a protocol-compliant
tiebreak, not a demonstrated superiority.

---

## 8. Relationship to the earlier `biodeg_gwu` study

[gwu.md](gwu.md) reports a descriptor-type study on `biodeg_gwu`, concluding that the
non-GWU set improves test ROC-AUC by **+0.0183** over baseline and recommending it as the
deployable configuration. That result must **not** be cited alongside the present work
without two qualifications:

1. **Different dataset.** `biodeg_gwu` retains inherently-biodegradable rows: 5742 train /
   **300** test, against 5264 / **278** here. Its test metrics are not on the 278-molecule
   test set used by the HGB and MPNN baselines, so they were never directly comparable to
   those figures.
2. **Test-selected.** Its feature-set ranking, and the preceding architecture sweep, were
   both decided on test ROC-AUC read from `agg/test/best.json`. This is the selection
   procedure that [porting guide §2](../dgt_porting_guide.md) explicitly prohibits.

Re-deriving the same comparison on validation data, on the corrected dataset, collapses the
descriptor effect from +0.0183 to +0.0001 on ROC-AUC. **This is itself a result**: it
quantifies how much of a reported ablation gain can be an artifact of selecting the winner on
the evaluation set, and it is a useful cautionary datapoint for the descriptor-fusion
literature.

---

## 9. Limitations and open items

Stated explicitly so that reviewers and future work are not misled.

1. **Single validation split, not cross-validation.** Selection used one fixed 90/10
   train/validation carve-out, not the stratified 5-fold cross-validation on train that
   [porting guide §2](../dgt_porting_guide.md) specifies. Consequences: (a) selection is
   noisier than the protocol intends; (b) the folds are not matched to the MPNN's, so the
   "same CV basis" requirement of §8 is not yet satisfiable and **no cross-model CV
   comparison is claimed here**. Implementing the §5 CV harness is the highest-value next
   step.
2. **The test set has been scored 16 times** (4 configurations × 4 seeds) as an automatic
   part of each run, though never read during selection. Only the selected configuration's
   test metric should be reported as a headline; the others are ablation context.
3. ~~Deployment seed is chosen by median test score.~~ **Resolved 2026-09-02.**
   `retrain_on_trainval.py` now selects the median seed on **validation**
   (`<seed>/val/stats.json`) and takes the retrain budget from that seed's best-validation
   epoch. It no longer opens anything under `<seed>/test/`, and the manifest records
   `seed_selected_on: validation`.
4. **Deployment threshold is optimised on test.** The F1-optimal decision threshold embedded
   in the deployment manifest is swept over test predictions. Any F1, precision, or recall
   quoted *at that threshold on the same test set* is optimistically biased. All headline
   metrics in this document and in the cross-model comparison use the fixed 0.5 threshold and
   are unaffected.
5. **Four seeds.** GPU training is non-deterministic; prior work on this endpoint observed
   run-to-run CV-F1 variation comparable to an entire hyperparameter grid's spread. Four
   seeds is the minimum for a credible ± and is insufficient to resolve differences of the
   size seen in §5.
6. **Architecture not tuned on this dataset.** Hyperparameters are inherited from a sweep run
   on `biodeg_gwu`, which was itself test-selected. No architecture search has been performed
   on `biodeg_gwu_no_ind`.

---

## 10. Reproducibility

Configurations, all in `configs/biodegradability/`:

| Feature set | Config |
|---|---|
| `none` | `BiodegNoInd-DGT-Pipeline.yaml` |
| `qm` | `BiodegNoInd-DGT-Pipeline-WithDesc-gwu.yaml` |
| `rdkit_fg` | `BiodegNoInd-DGT-Pipeline-WithDesc-nongwu.yaml` |
| `qm_rdkit` | `BiodegNoInd-DGT-Pipeline-WithDesc.yaml` |

```bash
# data snapshot (once, needs S3 credentials)
python scripts/prepare_data.py --dataset biodeg_gwu_no_ind \
  --trans-learn-path /home/jovyan/tools/trans_learn

# one configuration, 4 seeds
python main.py --cfg configs/biodegradability/<CONFIG>.yaml \
  --repeat 4 seed 0 wandb.use False optim.max_epoch 50

# selection on validation, test suppressed
python scripts/rank_configs_by_val.py \
  results/DGT/BiodegNoInd-DGT-Pipeline \
  results/DGT/BiodegNoInd-DGT-Pipeline-WithDesc-gwu \
  results/DGT/BiodegNoInd-DGT-Pipeline-WithDesc-nongwu \
  results/DGT/BiodegNoInd-DGT-Pipeline-WithDesc \
  --metric f1 --hide-test
```

Key implementation points: dataset loader
[`biodeg_gwu_no_ind.py`](../../graphgps/loader/dataset/biodeg_gwu_no_ind.py) (split
convention, train-only standardisation); training mode
[`dgt_train.py`](../../graphgps/train/dgt_train.py) (train+val loop, single final test pass);
descriptor head `line_graph_with_desc` in
[`san_graph.py`](../../graphgps/head/san_graph.py); selection utility
[`rank_configs_by_val.py`](../../scripts/rank_configs_by_val.py).

Environment: Python 3.10, PyTorch 2.1.0, PyTorch Geometric 2.0.4, RDKit 2025.9.1,
graph-tool 2.45. Exact package set in `environment.yaml`.

---

## 11. Remaining work before submission

- [x] Read the test set **once** for the selected configuration (§5.3, 2026-09-02).
- [x] Place those numbers against the HGB and MPNN references (§5.4).
- [ ] **AUPRC** — not yet computed for DGT; the baselines report it (HGB 0.9225 / 0.9249).
      `scripts/analyze_run.py` emits `average_precision` per seed; aggregate across the 4
      seeds to complete the comparison table.
- [ ] Implement the 5-fold CV harness (porting guide §5) so a matched cross-model CV
      comparison becomes possible; re-run selection under it.
- [ ] Increase seed count, or report CV mean ± std over folds, to resolve whether the
      descriptor channel has any real effect.
- [ ] Retrain the selected configuration on train+val and deposit the deployment bundle.
- [ ] Decide whether the "test-selected ablation gains do not replicate" finding (§8) is
      framed as a headline contribution or a methods note.
