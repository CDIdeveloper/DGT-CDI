# Framing options for the §8 finding — decision doc

**The question.** [paper.md](paper.md) §8 records that a descriptor-fusion gain reported as
**+0.0183 ROC-AUC** under test-based selection measures **+0.0035 (n.s.)** under 5-fold CV on
the corrected dataset. Is that the paper's headline, or a supporting methods note?

This document drafts both. Delete it once the decision is made and the chosen framing is
folded into paper.md.

---

## What the evidence actually supports

Four findings, in descending order of statistical strength:

| # | Finding | Strength |
|---|---|---|
| 1 | **QM descriptors are significantly weaker than RDKit/functional-group descriptors** and add nothing on top of them | ΔF1 −0.0091, **0/5 folds**, t = −7.36, p ≈ 0.002. The only significant effect in the study |
| 2 | **F1-argmax thresholds are unidentifiable** at n = 278 — plateau spans ~half the probability scale; argmax sits on a single molecule's score | Quantified on two independent checkpoints |
| 3 | **A single seed re-run reordered a four-arm ranking** and changed which configuration the protocol selects | Direct observation (§6.1) |
| 4 | **The +0.0183 descriptor gain does not replicate**: +0.0035, p ≈ 0.09 under CV | The headline candidate, but the comparison spans a dataset change *and* a protocol change |

Note on 4: the earlier study was on `biodeg_gwu` (5742/300, InD retained) and the present one
on `biodeg_gwu_no_ind` (5264/278). The collapse from +0.0183 to +0.0035 therefore confounds
two changes. **Findings 1–3 are cleaner than 4**, which matters for Option B.

---

## Option A — methods note

**Shape.** An application paper: DGT applied to ready-biodegradability, with a descriptor
ablation. §8 lives in the limitations/methods discussion, roughly a paragraph, explaining why
earlier internal numbers differ from the present ones. Findings 1–3 stay as observations
inside the results.

**Claim.** "DGT achieves F1 0.8610 ± 0.0066 / ROC-AUC 0.9196 ± 0.0027 on the 278-molecule
biodegradability test set; a descriptor-fusion channel provides no measurable benefit, and
quantum-mechanical descriptors specifically carry less signal than functional-group
descriptors."

**For**
- Submittable now. Nothing further to run.
- Single, clear contribution; straightforward to place in a cheminformatics/QSAR venue.
- The leak-free protocol reads as methodological rigour supporting the result, which is how
  most reviewers in the field will expect to see it.

**Against**
- Buries the most transferable content. The field publishes a steady stream of small ablation
  gains selected on test; you have quantified evidence about what happens to them.
- The substantive contribution is thin once stated plainly: the descriptor channel — the
  paper's own subject — does nothing, and the DGT-vs-baseline comparison (assembled elsewhere)
  is parity with gradient boosting, not a win.
- Findings 2 and 3 are genuinely useful and would go largely unread.

---

## Option B — headline contribution

**Shape.** A methods paper on **selection protocol in molecular property prediction**, using
biodegradability as the case study. Findings 1–4 are the results. The DGT numbers become the
worked example rather than the point.

**Claim.** "Ablation gains in molecular property prediction are routinely selected on the
evaluation set. We quantify the consequence on a biodegradability endpoint: a descriptor
channel reported at +0.0183 ROC-AUC under test-based selection measures +0.0035 (n.s.) under
stratified 5-fold cross-validation. We further show that (i) a single training-seed re-run
reorders a four-configuration ranking and changes the selected model, and (ii) F1-optimal
decision thresholds are unidentifiable at realistic test-set sizes, varying by 0.3 across
checkpoints whose ROC-AUC agrees to 3 × 10⁻⁵."

**For**
- Novel and transferable. Findings 2 and 3 are, as far as we know, not quantified anywhere in
  this literature, and both are cheap for others to check.
- Three mutually reinforcing results, one of them significant at p ≈ 0.002.
- The negative descriptor result becomes an asset rather than an anticlimax.
- Directly actionable: the tooling (`rank_configs_by_val.py`, `cv-train-<k>`, the
  null-threshold manifest) is a concrete protocol others can adopt.

**Against**
- **One dataset.** A methods claim on a single endpoint invites "does this generalise?" —
  reasonably. §12 item 6 (replicate on `biodeg` and `biodeg_gwu`) becomes a prerequisite, at
  roughly 4.3 h GPU per dataset plus analysis.
- Finding 4 confounds a dataset change with a protocol change (see above). Either disclose it
  prominently, or re-run the old `biodeg_gwu` ablation under CV to isolate the protocol effect
  — another ~4.3 h.
- Harder venue placement: falls between cheminformatics and ML-methodology.
- Requires framing the earlier internal study as the negative example, which needs care in
  how it is written.

---

## Option C — the pragmatic middle

Application paper, but findings 1–3 get a full results subsection rather than a footnote, and
the abstract mentions the selection-protocol result. §8 stays as-is: a documented,
quantified observation, not the headline.

Submittable now; the methodological content is discoverable and citable; leaves the door open
to a dedicated methods paper later once §12 item 6 has been done. Loses the framing punch of
Option B.

---

## Recommendation

**Option C now, Option B later — unless you are willing to spend ~9 h GPU plus analysis, in
which case go straight to B.**

The reasoning: B is the better paper and the more useful one, but it is not currently
supportable. A methods claim resting on one endpoint, where the flagship comparison also
confounds a dataset change, will draw exactly the objection you would draw yourself. The
missing work is well-defined and not large — replicate on two more datasets, and re-run the
old `biodeg_gwu` ablation under CV to separate protocol from dataset — but it is real work,
not a rewrite.

C banks the current result without foreclosing anything. If the extra datasets get run, the
methods paper is then supported by three endpoints instead of one, which is a much stronger
submission than B would be today.

**Against my own recommendation:** if the descriptor-fusion question is not one you intend to
return to, C risks publishing the weaker version of your most interesting finding and never
writing the stronger one. Papers deferred are frequently papers not written. If you know you
want B, the ~9 h is cheap for what it buys.
