# Upstream sync & fork provenance

> Durable reference for **maintenance**, not architecture. Answers two questions:
> *"which code is mine and which did I inherit?"* and *"how do I safely pull a future
> update from Shuyuan?"*
> For what DGT contributes **architecturally** over GraphGPS, see [tech.md](tech.md#integration-with-graphgps).

**Audited:** 2026-08-26, against `HEAD` on branch `mol-desc`.

---

## 1. Fork boundary

This repo was inherited from a colleague; it is not original work from `init`.

| | |
|---|---|
| Original author | **Shuyuan Zhang** `<sz469@cam.ac.uk>`, later `<zsy0016@gmail.com>` |
| Their commits | `7b5b72b` init (2025-11-01) · `fe79f7f` readme (2025-11-01) · `2e47729` feature fusion + chiro dataset (2026-05-03) · `ecf8f06` image updated (2026-05-03) |
| **Fork base** | **`ecf8f06`** — the last commit not authored by us |
| Our work begins | `65009d8` "understand model done" (2026-05-22, GZ82) — also where [CLAUDE.md](../CLAUDE.md) and `documents/` first appear |

`ecf8f06` is the anchor for every command in this document. Treat it as `FORK_BASE`.

```bash
FORK_BASE=ecf8f06
```

---

## 2. There are two fork layers — do not conflate them

```
GraphGPS  ──►  DGT (Shuyuan)  ──►  DGT-CDI (us)
   │              │                    │
   │              │                    └─ biodeg datasets, descriptor fusion,
   │              │                       dgt/dgt_retrain train modes, scripts/
   │              └─ DGTLayer, DGTModel, line_graph head, pairwise E tensors
   └─ GraphGym harness: registry, config system, train loop
```

[tech.md](tech.md#integration-with-graphgps) documents the **left** boundary
(GraphGPS → DGT). This file documents the **right** one (DGT → DGT-CDI). Both use
the same vocabulary — *shared / parallel alternative / additive registration* — which
is convenient but easy to misread. When tech.md says "upstream", it means GraphGPS.
When this file says "upstream", it means **Shuyuan's DGT**.

---

## 3. What is ours vs. inherited

### Entirely ours (directory did not exist at `FORK_BASE`)

- [scripts/](../scripts/) — `prepare_data.py`, `predict.py`, `analyze_run.py`,
  `_eval_plots.py`, `retrain_on_trainval.py`, `select_features_from_shap.py`
- [tests/](../tests/) — all of it, plus [pytest.ini](../pytest.ini)
- `documents/` — all of it
- [configs/biodegradability/](../configs/biodegradability/) — all 13 YAMLs

### Ours, but living inside their tree as **new files**

| File | What it is |
|---|---|
| [graphgps/train/dgt_train.py](../graphgps/train/dgt_train.py) | `@register_train('dgt')` — train+val loop, single final test pass |
| [graphgps/train/dgt_retrain.py](../graphgps/train/dgt_retrain.py) | `dgt_retrain` / `dgt_retrain_with_test` deployment modes |
| [graphgps/loader/dataset/biodeg_gwu.py](../graphgps/loader/dataset/biodeg_gwu.py) | biodeg GWU batch-2 PyG dataset |
| [graphgps/loader/dataset/biodeg.py](../graphgps/loader/dataset/biodeg.py) | biodeg no-Reaxys PyG dataset |
| [graphgps/loader/dataset/_mol_featurise.py](../graphgps/loader/dataset/_mol_featurise.py) | shared `smiles_to_xy` featurisation |
| [graphgps/loader/dataset/_desc_select.py](../graphgps/loader/dataset/_desc_select.py) | descriptor-column selection + cache hash keying |

**The convention is additive registration, not "new code goes in `scripts/`."** New train
modes register via `@register_train` *alongside* upstream `custom` rather than editing it;
new datasets register through `master_loader`; YAML strings select which one runs. That is
why the DGT model core is untouched (§4).

### Untouched inherited core

`graphgps/layer/dgt_layer.py`, `graphgps/network/dgt_model.py`, every file in
`graphgps/encoder/` except one, and all GPS/SAN sibling baselines are **byte-identical**
to `FORK_BASE`. A future upstream update touching only these will merge cleanly.

---

## 4. Inherited files we modified — conflict risk on merge

Eight files. Ordered by merge risk, highest first.

| File | Δ | What we changed | Risk |
|---|---|---|---|
| [graphgps/loader/master_loader.py](../graphgps/loader/master_loader.py) | +63 −0 | `preformat_Biodeg` / `preformat_BiodegGwu` appended after `preformat_QM9`; **plus 2 mid-file hunks**: an import at ~L15 and a dispatch branch at ~L149 | **Med** — mid-file hunks are where a conflict will land |
| [graphgps/head/san_graph.py](../graphgps/head/san_graph.py) | +75 −0 | `LineGraphWithDescHead` appended after `LineGraphHead` | **Med** — it's *their* file; a new head would have been cleaner as a new file in `graphgps/head/` |
| [graphgps/transform/transforms.py](../graphgps/transform/transforms.py) | +12 −2 | graph_tool compat shim: `remove_self_loops`/`remove_parallel_edges` moved from `graph_tool.stats` to `graph_tool.generation` in newer builds; wrapped in try/except (commit `a887dc3`) | **Low** — env bugfix; drop it if upstream fixes it themselves |
| [graphgps/config/dataset_config.py](../graphgps/config/dataset_config.py) | +19 −0 | `desc_dim`, `standardize_desc`, `desc_include/exclude/columns` fields, appended | Low |
| [graphgps/encoder/linear_edge_encoder.py](../graphgps/encoder/linear_edge_encoder.py) | +7 −0 | two `elif` branches for `PyG-biodeg_gwu` / `PyG-biodeg` | Low |
| [graphgps/config/custom_gnn_config.py](../graphgps/config/custom_gnn_config.py) | +6 −0 | `gnn.desc_proj_dim`, appended | Low |
| [main.py](../main.py) | +5 −0 | total-runtime log line | Low |
| [graphgps/\_\_init\_\_.py](../graphgps/__init__.py) | +2 −2 | commented out `from .pooling import *` and `from .stage import *` | Low — see note |

Also: `imgs/dgt.png` replaced (binary), and `chiro3d_molecule_net.py` → `chiral3d_molecule_net.py`
in `727b0a3` (git records `R100`, a pure rename, zero content change).

**Note on `graphgps/__init__.py`.** `graphgps/pooling/` and `graphgps/stage/` **never existed**
in the inherited tree — verified with `git ls-tree -r --name-only $FORK_BASE | grep -E 'pooling|stage'`
(no output). Those were dead imports that would raise on any `import graphgps`, and nothing in the
repo calls `register_pooling` or `register_stage`. If a future upstream update *adds* those
directories, uncomment the two lines.

Every other edit is **purely additive** (zero deleted lines). A `git diff $FORK_BASE HEAD`
reads as pure surface area, which was the intent.

---

## 5. Fetching an update from Shuyuan

**There is currently no remote for their repo.** `origin` is `CDIdeveloper/DGT-CDI` — ours.
Their repo URL is not recorded anywhere in this checkout; get it from Shuyuan before step 1.

```bash
# 1. one-time: add their repo as a second remote
git remote add upstream <SHUYUAN_REPO_URL>
git fetch upstream

# 2. see what actually moved since our fork base
git log --oneline $FORK_BASE..upstream/main
git diff --stat $FORK_BASE upstream/main

# 3. does it touch any of our 8 modified files?  <-- the only real question
git diff --name-only $FORK_BASE upstream/main | grep -Ff <(
  git diff --name-only --diff-filter=M $FORK_BASE HEAD
)
```

If step 3 prints nothing, the merge is mechanical. If it prints `master_loader.py` or
`san_graph.py`, expect to resolve by hand — our hunks are appended blocks, so keep both
sides rather than taking either wholesale.

**Then:**

- [ ] Merge on `main`, not on a feature branch. As of this audit `mol-desc` is still
      unmerged (see [session_state.md](session_state.md)) — land or rebase that first,
      or you will be resolving the same conflicts twice.
- [ ] Re-run `pytest` — [tests/test_dataset.py](../tests/test_dataset.py) and
      [tests/test_descriptor_head.py](../tests/test_descriptor_head.py) cover the
      seams where our code meets theirs.
- [ ] Re-check the registry keys still resolve: `dgt` / `dgt_retrain` in `train_dict`,
      `line_graph_with_desc` in the head registry, `PyG-biodeg{,_gwu}` in `master_loader`.
      A silent registry regression shows up as a config-key error at startup, not a test failure.
- [ ] If their update changes featurisation, encoders, or anything upstream of `batch.desc`,
      **delete `datasets/*/processed/`** and rebuild — the processed cache is keyed on
      descriptor selection only, never on code version, so it will not self-invalidate.
- [ ] Update the `FORK_BASE` anchor and the §4 table in this file.

---

## 6. Reproducing this audit

```bash
FORK_BASE=ecf8f06

# who wrote what
git log --format="%h | %an <%ae> | %ad | %s" --date=short | tail -8

# every inherited file we touched, with add/delete counts
git diff --numstat $FORK_BASE HEAD -- $(git ls-tree -r --name-only $FORK_BASE | tr '\n' ' ')

# everything we added vs modified under graphgps/
git diff --name-status $FORK_BASE HEAD -- graphgps/ | sort

# where our hunks land inside their files (conflict forecast)
git diff $FORK_BASE HEAD -- graphgps/ main.py | grep -E "^@@|^\+\+\+"
```
