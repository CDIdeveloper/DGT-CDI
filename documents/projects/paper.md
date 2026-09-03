# DGT on biodegradability (`biodeg_gwu_no_ind`) — paper base

Working document for a publication on applying the Dual Graph Transformer (DGT) to
ready-biodegradability classification, with a molecular-descriptor late-fusion ablation.

**Status (2026-09-02):** configuration selected on validation with test suppressed; test set
then read **once** for the selected configuration (§5.3). Headline: DGT reaches
**F1 0.8610 ± 0.0066, ROC-AUC 0.9196 ± 0.0027** on the 278-molecule test set.

Related: [gwu.md](gwu.md) (earlier study on the *different* `biodeg_gwu` dataset —
see §8 before citing it), [../dgt_porting_guide.md](../dgt_porting_guide.md) (protocol
this work follows), [../trained_models.md](../trained_models.md) (model registry).

---

## 1. Contribution

Prior work on this endpoint indicated that descriptor-based gradient boosting outperforms a
message-passing neural network, suggesting the endpoint is driven by composition and
functional-group content that a purely graph-based view under-represents. **This work asks
whether a graph transformer with an explicit descriptor-fusion channel performs
competitively**, and — separately — whether the descriptor channel contributes at all once
model selection is made without reference to the test set.

The second question turns out to matter: a previously reported descriptor benefit on a
related dataset does not survive re-derivation on validation data (§8).

**Scope.** This document covers the DGT results produced in this repository. Cross-model
comparison against the gradient-boosting and MPNN baselines is assembled centrally, where
those models' full provenance is available; no baseline numbers are reproduced here (§5.4).

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
   used for the baseline models, so the numbers remain comparable when assembled centrally.
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

Two items were initially *not* satisfied by the deployment path; both have since been
**fixed in code**. Seed choice and epoch budget are now validation-derived (§9 item 3), and
the decision threshold is now fitted on validation predictions rather than swept on test
(§9 item 4). The runs behind §5.3 predate the threshold fix and therefore still carry a
test-derived threshold in their manifest — this affects only
`predict.py --threshold optimal-f1`, never the 0.5-threshold metrics reported here.

---

## 5. Results — validation

Four seeds per configuration; mean ± population standard deviation across seeds. Each seed
contributes the metric at its own best-validation epoch.

### 5.1 Summary

| Feature set | desc_dim | Val F1 | Val ROC-AUC | Median best epoch (F1 / AUC) |
|---|---|---|---|---|
| `qm_rdkit` (all) | 247 | **0.8165 ± 0.0037** | 0.8853 ± 0.0026 | 32 / 32 |
| `rdkit_fg` (non-GWU) | 207 | 0.8164 ± 0.0065 | **0.8876 ± 0.0010** | 40 / 29 |
| `none` (graph only) | — | 0.8147 ± 0.0054 | **0.8900 ± 0.0037** | 35 / 27 |
| `qm` (GWU only) | 40 | 0.8119 ± 0.0050 | 0.8829 ± 0.0015 | 37 / 32 |

> **Graph-only arm re-run 2026-09-02.** Its seed 0 was overwritten by a smoke test (§10) and
> re-trained for 50 epochs. F1 figures above are the refreshed values; the ROC-AUC figure is
> pending a re-ranking. The re-run moved seed 0's val F1 from 0.8053 to 0.8180 — **0.0127 on a
> single seed from GPU non-determinism alone**, which is larger than every gap in this table
> and is the strongest single piece of evidence for §7's conclusion that these four
> configurations are not distinguishable at four seeds. It also **reordered the ROC-AUC
> ranking**: the graph-only arm moved from second to first (0.8875 → 0.8900). See §6 for how
> the recorded selection is handled in light of this.

### 5.2 Per-seed values

| Feature set | Val F1 (seeds 0–3) | Val ROC-AUC (seeds 0–3) |
|---|---|---|
| `none` (re-run) | 0.8180, 0.8061, 0.8144, 0.8202 | 0.8912†, 0.8932, 0.8919, 0.8838 |
| `qm` | 0.8124, 0.8137, 0.8175, 0.8039 | 0.8808, 0.8845, 0.8843, 0.8822 |
| `rdkit_fg` | 0.8202, 0.8121, 0.8084, 0.8249 | 0.8892, 0.8866, 0.8872, 0.8874 |
| `qm_rdkit` | 0.8164, 0.8106, 0.8180, 0.8208 | 0.8865, 0.8877, 0.8810, 0.8859 |

† Seed 0's re-run value is derived from the refreshed 4-seed mean rather than read directly;
seeds 1–3 are unchanged. Pre-re-run seed 0 was F1 0.8053 / ROC-AUC 0.8812.

### 5.2b Cross-validation — stratified 5-fold on train

The single-split figures above could not order the four arms (§6.1), so the selection was
re-derived under the protocol [porting guide §2](../dgt_porting_guide.md) specifies: stratified
5-fold on the **train** split (`random_state=1`), a fresh model per fold, the held-out fold as
validation, the fixed 278-row test split untouched by every fold. Each fold evaluates on
**1053 molecules against the single split's 526**, and every training molecule is held out
exactly once (fold sizes 1053×4 + 1052 = 5264; fold positives summing to 2466).

| Rank | Feature set | CV F1 (mean ± std) | CV ROC-AUC (mean ± std) | Median best epoch |
|---|---|---|---|---|
| 1 | `rdkit_fg` (non-GWU, 207) ← **selected** | 0.8143 ± 0.0100 | **0.8928 ± 0.0065** | 28 |
| 2 | `qm_rdkit` (all, 247) | 0.8142 ± 0.0129 | 0.8925 ± 0.0064 | 27 |
| 3 | `none` (graph only) | 0.8134 ± 0.0060 | 0.8893 ± 0.0051 | 32 |
| 4 | `qm` (GWU only, 40) | 0.8052 ± 0.0083 | 0.8887 ± 0.0052 | 30 |

Per-fold values are in `results/DGT_cv/dgt_cv_results.{json,md}`.

**Dispersion is larger under CV than across seeds** — fold std 0.0060–0.0129 on F1 against
0.0037–0.0065 for the 4-seed single-split runs. That is expected and is the point: fold
variation measures sensitivity to the *data split*, which exceeds optimisation noise. It is
the honest dispersion for this quantity.

**The four arms remain statistically indistinguishable.** The F1 range across all four is
0.0091 — smaller than the fold std of two of them. The top three lie within 0.0009 of each
other.

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
| **AUPRC** (average precision) | **0.9269 ± 0.0051** |

Per-seed AUPRC: 0.9340, 0.9280, 0.9260, 0.9198.

### 5.4 Notes for cross-model comparison

**Out of scope for this document.** Comparison against the gradient-boosting and MPNN
baselines is performed centrally, where those models' full provenance (selection protocol,
dispersion, package versions) is available. This repository owns the DGT numbers only.

What to carry forward when the comparison is assembled:

- All figures in §5.3 are at the **fixed 0.5 threshold**, from probabilities, on the fixed
  278-molecule test split — the definitions required for comparability.
- DGT was selected on a **single 90/10 validation split**, not 5-fold CV on train (§9 item 1).
  A CV-matched comparison is not yet possible from this side; implementing the CV harness is
  the open item that would enable it.
- **AUPRC is not yet computed** for DGT (§11).
- Dispersion is over **4 training seeds** on a fixed split — it captures optimisation
  variance, not sampling variance of the 278-molecule test set, which is substantially
  larger. Seed std should not be read as a confidence interval on the metric.

---

## 6. Selection decision

Applying the §2 rule, in order, before any test number was viewed:

1. **F1 primary.** `qm_rdkit` (0.8165 ± 0.0037) vs `rdkit_fg` (0.8164 ± 0.0065). The
   difference is 0.0001 — an order of magnitude inside either configuration's seed standard
   deviation. Declared a tie.
2. **Tiebreak on ROC-AUC.** `rdkit_fg` (0.8876) > `qm_rdkit` (0.8853).
3. **Selected: `rdkit_fg` (non-GWU, 207 descriptors).**

This decision, including the fact that the tiebreak was invoked, was recorded prior to
reading the test set, as [porting guide §2](../dgt_porting_guide.md) requires. It is the
selection this work reports.

### 6.1 The selection is not stable — and this is disclosed, not corrected

After the test set had been read, the graph-only arm's seed 0 was re-trained (its artifacts
had been destroyed by an unrelated smoke test, §10). On the refreshed numbers the picture
changes materially:

- **All four arms tie on F1.** Gaps from the leader are +0.0000, +0.0001, +0.0018, +0.0046,
  against seed standard deviations of 0.0037–0.0065. Every arm falls inside the tie band.
- **The ROC-AUC tiebreak then selects `none` (graph-only), not `rdkit_fg`** — the graph-only
  arm rose from 0.8875 to 0.8900 and took first place.

So a single seed re-run, of an arm that was not even in contention, changed which
configuration the protocol selects.

**This is reported rather than acted on.** Re-running the selection now — after the test set
has been read — would not be the pre-registered decision that §4 item 1 claims, so the
recorded selection stands. Note that switching would move to the arm with *worse* test
performance (F1 0.8087 vs 0.8610), so the choice to leave it is not test-motivated in either
direction.

The substantive conclusion is §7's: **at four seeds on a single validation split, these four
configurations cannot be ordered.** The recorded selection is a protocol-compliant tiebreak
among indistinguishable candidates, not evidence that the descriptor channel helps.

### 6.2 Cross-validation confirms the recorded selection

The instability above was the motivation for running the §2 cross-validation protocol
(§5.2b). Under 5-fold CV, applying the identical rule:

1. **All four arms tie on F1** (range 0.0091, inside two arms' fold std).
2. **Tiebreak on ROC-AUC → `rdkit_fg`** (0.8928), ahead of `qm_rdkit` (0.8925),
   `none` (0.8893) and `qm` (0.8887).
3. **Selected: `rdkit_fg`** — the same configuration recorded in §6.

This is the outcome that matters for the reported result. The single-split selection was made
under clean conditions but turned out to be unstable; the CV selection, made on ten times the
validation data with dispersion measured across data splits rather than seeds, **independently
reproduces it**. The graph-only arm that briefly took first place on the single-split re-run
falls to third under CV. The recorded selection therefore stands on a sound basis, not merely
on the technicality of having been written down first.

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

**The model is well calibrated at 0.5.** Scoring the deployment model — retrained on
train+val, with the 278 test molecules still held out — gives an F1-optimal threshold of
**0.5013**, indistinguishable from the default 0.5. No threshold tuning is required, and the
0.5-thresholded metrics reported throughout are therefore also near-optimal rather than an
arbitrary operating point. Notably, the F1-optimal threshold measured on an *individual
training seed's* model was 0.375: threshold estimates do not transfer between checkpoints
trained on different data, which is an argument for deriving any deployment threshold from
the shipped artifact itself rather than from a training run.

**The four configurations are not distinguishable, and cross-validation confirms it rather
than resolving it.** Under 5-fold CV the F1 range across all four arms is 0.0091, smaller than
the fold standard deviation of two of them, and the top three lie within 0.0009. Testing the
descriptor channel directly — `rdkit_fg` minus `none`, paired fold by fold:

| Metric | Per-fold difference | Mean | Folds favouring descriptors |
|---|---|---|---|
| F1 | −0.0040, −0.0110, +0.0176, +0.0014, +0.0006 | **+0.0009** | 3 / 5 |
| ROC-AUC | +0.0039, −0.0015, +0.0050, +0.0080, +0.0021 | **+0.0035** | 4 / 5 |

On F1 the descriptor channel does nothing (t = 0.20, df = 4). On ROC-AUC it is consistently
but weakly positive (t = 2.22, df = 4; p ≈ 0.09) — suggestive, short of conventional
significance at five folds, and not corrected for having compared four arms. The honest
statement is that **any descriptor benefit on this endpoint is at most ~0.004 ROC-AUC and
cannot be established at this sample size** — against the +0.0183 reported by the earlier
test-selected study (§8).

---

## 8. Relationship to the earlier `biodeg_gwu` study

[gwu.md](gwu.md) reports a descriptor-type study on `biodeg_gwu`, concluding that the
non-GWU set improves test ROC-AUC by **+0.0183** over baseline and recommending it as the
deployable configuration. That result must **not** be cited alongside the present work
without two qualifications:

1. **Different dataset.** `biodeg_gwu` retains inherently-biodegradable rows: 5742 train /
   **300** test, against 5264 / **278** here. Its test metrics are not on the 278-molecule
   test set used by the baseline models, so they were never directly comparable to those
   figures.
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

1. ~~Single validation split, not cross-validation.~~ **Resolved 2026-09-02.** The §2
   protocol is implemented (`split_mode: cv-train-<k>` plus `scripts/cv/`) and the selection
   re-derived under stratified 5-fold CV on train with `random_state=1` (§5.2b, §6.2). Folds
   are built to the guide's specification, so a CV-matched cross-model comparison is now
   possible from this side. **Residual caveat:** the CV runs use one training seed per fold,
   so fold dispersion mixes data-split variation with optimisation noise; separating them
   would need seeds within folds.
2. **The test set has been scored 16 times** (4 configurations × 4 seeds) as an automatic
   part of each run, though never read during selection. Only the selected configuration's
   test metric should be reported as a headline; the others are ablation context.
3. ~~Deployment seed is chosen by median test score.~~ **Resolved 2026-09-02.**
   `retrain_on_trainval.py` now selects the median seed on **validation**
   (`<seed>/val/stats.json`) and takes the retrain budget from that seed's best-validation
   epoch. It no longer opens anything under `<seed>/test/`, and the manifest records
   `seed_selected_on: validation`.
4. **Deployment threshold: fixed in code, but the shipped bundle predates the fix.**
   `analyze_run.py` now fits the F1-optimal decision threshold on **validation** predictions
   and applies it to test, and `dgt_train.py` dumps `<seed>/val/predictions.pt` to enable
   this. Runs trained before that change lack the file and fall back to sweeping on test,
   with an explicit warning. The 4 seeds behind §5.3 are such runs, so the threshold in the
   current deployment manifest is test-derived. This affects **only**
   `predict.py --threshold optimal-f1`; every metric reported in this document uses the fixed
   0.5 threshold and is unaffected. Re-running the four seeds would produce a
   validation-fitted threshold, at the cost of new (non-identical) test numbers, since GPU
   training is not bit-reproducible. **In practice this turned out to be moot**: the deployed
   model's own F1-optimal threshold is 0.5013 (§7), so 0.5 is the correct operating point and
   no fitted threshold is needed.

   **Cost asymmetry.** F1 optimisation weights false positives and false negatives equally.
   For this endpoint the positive class is *readily biodegradable*, so a false positive means
   wrongly clearing a persistent compound — plausibly the more costly error in a screening
   application. Any deployment threshold should be chosen from the application's error costs
   (e.g. a target precision on the positive class), not from maximum F1. This work reports at
   0.5 and makes no operating-point recommendation.
5. **Four seeds.** GPU training is non-deterministic; prior work on this endpoint observed
   run-to-run CV-F1 variation comparable to an entire hyperparameter grid's spread. Four
   seeds is the minimum for a credible ± and is insufficient to resolve differences of the
   size seen in §5.
6. **Descriptor standardisation is not refit per CV fold.** The z-score mean/σ are computed
   inside the dataset loader's `process()` from the train-parquet rows excluding the original
   fixed 10 % carve-out, and are baked into the processed cache — they do not change with
   `split_mode`. Under `cv-train-5` this means each fold's validation rows contributed to the
   normalisation constants that are then applied to them. Porting guide §7 item 3 requires
   scalers to be fit on the **training folds** during CV, so this is a genuine deviation.
   Magnitude is small: each molecule contributes ~1/4738 to a mean, and the **test rows never
   contributed at all**, so the reported test metrics are unaffected. Direction matters for
   interpretation — only the three descriptor arms are touched (the graph-only arm applies no
   standardisation), and the effect very slightly *favours* them, so a null descriptor result
   under CV is if anything understated. Fixing it properly requires either a per-fold
   processed cache (5× featurisation cost) or moving standardisation out of the loader into a
   post-split transform; deferred.

7. **Architecture not tuned on this dataset.** Hyperparameters are inherited from a sweep run
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

**Artifact note (2026-09-02).** A 3-epoch smoke test was run against
`BiodegNoInd-DGT-Pipeline` after the ablation completed. `main.py` wipes and recreates the
per-seed directory on each run, so the graph-only arm's **seed-0 artifacts on disk were
overwritten** by that short run; its `agg/` is consequently inconsistent. The per-seed values
in §5.2 are the originals and are what the §6 selection was made on — they are not re-derived
from the current on-disk state. The selected configuration
(`BiodegNoInd-DGT-Pipeline-WithDesc-nongwu`) is unaffected; all four of its seeds are intact.
Re-running the graph-only arm would produce different numbers (GPU training is not
bit-reproducible) and would no longer represent the basis on which the selection was made, so
the originals are retained.

---

## 10.1 Deployed artifacts

Two models were retrained on train+val (`dgt_retrain`; test held out) and deposited. Both are
on **`biodeg_gwu_no_ind`** (5264 train / 278 test). Bundles are
`final_model.{ckpt,config.yaml,json}` plus a `deploy_eval/` folder holding the ROC / PR /
confusion / score-histogram curves and per-molecule scores, so a downstream user can pick a
different operating point from their own error costs.

S3 prefix: `s3://cdi-lab-workspaces/ts_project_1/models/biodegradation/GWU/`

| | `biodeg-no-ind-dgt-nongwu-2026-09-02` | `biodeg-no-ind-dgt-graphonly-2026-09-02` |
|---|---|---|
| Feature set | `rdkit_fg` (207 descriptors) | `none` (graph only) |
| Config | `BiodegNoInd-DGT-Pipeline-WithDesc-nongwu.yaml` | `BiodegNoInd-DGT-Pipeline.yaml` |
| Basis for shipping | the **recorded selection** (§6) | **operability** — see below |
| Inference input | SMILES **+ 207 descriptor columns** | **SMILES only** |
| Seed / retrain budget | 2 / 22 epochs | 2 / 29 epochs |
| Deployed-model ROC-AUC | 0.9189 | 0.8975 |
| Deployed-model AUPRC | 0.9295 | 0.9058 |
| Deployed-model F1 | 0.8746 @ thr 0.5013 | 0.8322 @ thr 0.5475 |
| Decision threshold | 0.5 (measured optimum 0.5013) | 0.5475 |

Deployed-model figures are **single-checkpoint point estimates** on the 278 test molecules,
which `dgt_retrain` held out. They are not the generalisation estimate — that is §5.3's
4-seed mean ± std — and they are not interchangeable with it.

**Why the graph-only model is also deployed.** It requires only SMILES at inference, whereas
the descriptor model obliges every caller to compute and supply 207 RDKit/functional-group
columns in the exact training order. Given that §6.1 finds the two indistinguishable on
validation, that is a large operational simplification for no established loss of quality.
This is an engineering decision, **not** a claim that the graph-only model is better.

Thresholds were measured on each deployed checkpoint rather than inherited from a training
seed — the two differ (0.5013 vs 0.5475), and neither matches the 0.375 that a training
seed's model produced (§7).

**These artifacts are provisional.** If the cross-validation protocol (§9 item 1, §11) yields
a better-supported configuration, new bundles will be uploaded under a new dated model name;
the prefixes above are not overwritten.

---

## 11. Remaining work before submission

- [x] Read the test set **once** for the selected configuration (§5.3, 2026-09-02).
- [x] **AUPRC** — 0.9269 ± 0.0051 (§5.3, 2026-09-02).
- [x] **5-fold CV harness** (porting guide §5) implemented and run; selection re-derived and
      confirmed (§5.2b, §6.2, 2026-09-02).
- [ ] The descriptor effect remains unresolved: ROC-AUC +0.0035 paired across folds
      (p ≈ 0.09, 4/5 folds). Either accept "no established benefit" as the finding, or add
      seeds within folds / more folds to settle it.
- [x] Retrain the selected configuration on train+val and deposit the deployment bundle
      (§10.1). Optional refinement: the bundle's 22-epoch budget came from one seed's val
      curve; CV puts the median best epoch at 28.
- [ ] Decide whether the "test-selected ablation gains do not replicate" finding (§8) is
      framed as a headline contribution or a methods note.
