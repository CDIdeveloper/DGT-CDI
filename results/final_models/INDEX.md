# Final models — index

Every deployment bundle produced by this repo. **Bundles are never overwritten**: a new
checkpoint gets a new dated directory, and superseded ones are retained as the record.

A bundle is `final_model.{ckpt,config.yaml,json}` — weights, the architecture needed to
rebuild the model before loading them, and a manifest carrying provenance plus (for descriptor
models) the descriptor column contract. Newer bundles also carry `deploy_eval/` with ROC / PR
/ confusion curves and per-molecule test scores.

S3 root: `s3://cdi-lab-workspaces/ts_project_1/models/biodegradation/GWU/`

---

## Current — `biodeg_gwu_no_ind` (5264 train / 278 test)

The canonical dataset ([dgt_porting_guide.md §1](../../documents/dgt_porting_guide.md)).
Configuration selected on **validation with test suppressed**, then confirmed by 5-fold CV
([projects/paper.md](../../documents/projects/paper.md) §5.2b, §6.2). Full write-up in
paper.md §10.1.

| Bundle | Feature set | desc_dim | Retrain budget | Threshold | Inference input | S3 |
|---|---|---|---|---|---|---|
| `biodeg-no-ind-dgt-nongwu-2026-09-03` | `rdkit_fg` | 207 | 29 ep (CV median 28 + 1) | 0.5 | SMILES **+ 207 descriptor columns** | ✅ 9 obj |
| `biodeg-no-ind-dgt-graphonly-2026-09-03` | `none` | — | 33 ep (CV median 32 + 1) | 0.5 | **SMILES only** | ✅ 9 obj |

Test metrics (single checkpoint, on the 278 molecules `dgt_retrain` held out — **not** the
reported generalisation estimate, which is the 4-seed mean ± std in paper.md §5.3):

| Bundle | ROC-AUC | AUPRC | F1 @ 0.5 | Precision | Recall |
|---|---|---|---|---|---|
| `...-nongwu-2026-09-03` | 0.9198 | 0.9333 | 0.8649 | 0.8421 | 0.8889 |
| `...-graphonly-2026-09-03` | 0.8975 | 0.9085 | 0.8276 | 0.8219 | 0.8333 |

**Which to deploy.** The two are statistically indistinguishable on this dataset (paper.md
§6.2, §7). `graphonly` needs only SMILES; `nongwu` obliges the caller to compute and supply
207 RDKit/functional-group columns in training order. Unless you need the descriptor variant
specifically, **deploy `graphonly`**.

**Threshold.** Both ship 0.5. F1-argmax thresholds are not identifiable at n = 278 — the F1
surface is flat over roughly half the probability scale — so the argmax is noise. Each
manifest keeps its argmax under `best_f1_threshold_argmax_on_test`. To pick a different
operating point, use `deploy_eval/pr.png` and the per-molecule scores; for a screening
application choose a target precision on the readily-biodegradable class rather than max F1.
Full analysis: paper.md §7.

## Superseded — retained as record

| Bundle | Why superseded | S3 |
|---|---|---|
| `biodeg-no-ind-dgt-nongwu-2026-09-02` | 22-epoch budget from one seed's validation curve; replaced by the CV-derived 29 | ✅ 9 obj |
| `biodeg-no-ind-dgt-graphonly-2026-09-02` | 29-epoch budget from one seed; replaced by the CV-derived 33. Also still carries the F1-argmax threshold 0.5475 | ✅ 9 obj |

S3 state verified 2026-09-03. A complete bundle is **9 objects**; the three legacy bundles
below were uploaded under earlier conventions and are not counted here.

## Legacy — pre-protocol, do not cite

⚠️ These three predate the validation-based selection protocol. Their manifests show
`seed_selected_on` absent, meaning the deployment seed was chosen by **median test metric**,
and their thresholds are F1-argmax on test — note 0.188 and 0.788 for near-identical model
families, which is the plateau-noise effect quantified in paper.md §7. They are also on
**different datasets** from the current bundles and are not comparable to them.

| Bundle | Dataset | Config | Budget | Seed | Threshold |
|---|---|---|---|---|---|
| `Biodeg-DGT-Pipeline` | `biodeg` (no-Reaxys) | `Biodeg-DGT-Pipeline` | 35 ep | 2 | 0.7881 |
| `Biodeg-GWU-DGT-Pipeline` | `biodeg_gwu` (300-row test) | `Biodeg-GWU-DGT-Pipeline` | 27 ep | 1 | 0.1876 |
| `Biodeg_GWU-DGT-Pineline-WithDesc-nongwu` | `biodeg_gwu` (300-row test) | `Biodeg-GWU-DGT-Pipeline-WithDesc-nongwu` | 26 ep | 1 | 0.5308 |

The third directory name contains a typo (`Pineline`); left as-is because renaming would break
any external reference to the path.

---

## Conventions

- **Naming:** `<dataset-slug>-dgt-<feature-set>-<YYYY-MM-DD>`. The directory name is also the
  S3 prefix, so local and cloud stay aligned.
- **Never overwrite.** New checkpoint → new dated directory. Manifest-only corrections may be
  refreshed in place, since the weights are unchanged.
- **Upload with `sync --delete`, not `cp --recursive`.** `cp --recursive` only adds and
  overwrites, so re-uploading a bundle whose `deploy_eval/` contents were *renamed* leaves the
  old files behind. This happened to `biodeg-no-ind-dgt-nongwu-2026-09-03`, which accumulated
  15 objects against the expected 9. A complete bundle is **9 objects**: 3 bundle files plus 6
  in `deploy_eval/` (roc, pr, confusion, score_hist, summary.json, scores CSV).
  ```bash
  aws s3 sync <bundle> "${S3ROOT}/<bundle>/" --delete --dryrun   # inspect first
  aws s3 sync <bundle> "${S3ROOT}/<bundle>/" --delete
  ```
- **`sync` skips same-size files — `cp` the metrics file explicitly afterwards.** `aws s3 sync`
  compares size and mtime, so a changed file whose size coincidentally matches the remote is
  silently skipped. This happened with `summary.json`: the v1 and v2 versions are both 562
  bytes, so a `sync` that correctly replaced all four PNGs left the **wrong metrics file** in
  S3 — and the object count still came out at the expected 9, so a count check would have
  passed. `cp -a` makes it worse by preserving the old mtime. After any `sync`:
  ```bash
  aws s3 cp <bundle>/deploy_eval/summary.json "${S3ROOT}/<bundle>/deploy_eval/summary.json"
  aws s3 cp "${S3ROOT}/<bundle>/deploy_eval/summary.json" - \
    | python -c "import json,sys;d=json.load(sys.stdin);print(d['roc_auc'],d['best_f1_threshold'])"
  ```
  Read it back from S3 and check the numbers. Object counts do not verify content.
- **Refreshing `deploy_eval/`: delete it first.** `cp -r SRC DEST` copies *into* `DEST` when
  `DEST` already exists, producing a nested `deploy_eval/<src-name>/` and leaving the previous
  checkpoint's curves at the top level — where they read as if they described the shipped
  model. This happened once and put the v1 model's PR curve in a v2 bundle. Always:
  ```bash
  rm -rf <run_dir>/deploy_eval
  cp -r /tmp/<scores>_eval <run_dir>/deploy_eval
  cp    /tmp/<scores>.csv  <run_dir>/deploy_eval/
  ls -1 <run_dir>/deploy_eval    # expect 6 files, no subdirectory
  ```
- **Local vs S3 may differ.** A bundle present here is not necessarily uploaded. The shared
  `GWU/` prefix also holds other projects' models, so list the specific bundle prefixes rather
  than the whole prefix:
  ```bash
  S3ROOT=s3://cdi-lab-workspaces/ts_project_1/models/biodegradation/GWU
  for M in $(ls -d */ | tr -d /); do
    n=$(aws s3 ls "${S3ROOT}/${M}/" --recursive 2>/dev/null | wc -l | tr -d ' ')
    printf '%-45s %s objects in S3\n' "$M" "$n"
  done
  ```
  A count of 0 means local-only. Each uploaded bundle should show at least 3 objects
  (`final_model.ckpt`, `.config.yaml`, `.json`), plus `deploy_eval/` contents where present.
- **Reading a manifest:**
  ```bash
  python -c "import json;print(json.dumps(json.load(open('<bundle>/final_model.json')),indent=2))"
  ```
- **Scoring a bundle** on a labelled table (this is also how a threshold is measured — always
  on the checkpoint being shipped, never inherited from a training seed):
  ```bash
  python scripts/predict.py --ckpt <bundle>/final_model.ckpt \
    --smiles-csv <table> --label-col <target> --output-csv /tmp/scores.csv
  ```
