"""Inference on new SMILES with a trained DGT model.

Supports both binary-classification (`task_type: classification_binary`) and
single-target regression (`task_type: regression`) checkpoints. The task type
is read from the bundled config — no CLI flag needed.

CLI
---
    python scripts/predict.py \\
        --ckpt        <path/to/final_model.ckpt>     (required)
        --smiles-csv  <input.csv|.parquet|s3://...>  (required; csv/parquet, local or s3)
        --output-csv  <output.csv>                   (required)
        [--orig-config <path>]                       (default: auto-discover)
        [--threshold  0.5 | optimal-f1]              (classification-only;
                                                      default 0.5; ignored
                                                      for regression)
        [--batch-size 64]                            (default: 64)
        [--smiles-col smiles]                        (default: 'smiles')
        [--label-col <col>]                          (optional; enables metrics)
        [--plot-dir <dir>]                           (default: <out>_eval/)
        [--no-plots]                                 (metrics only, with --label-col)

Inference flow
--------------
1.  Resolve the *pristine* YAML config (the dumped <run_dir>/config.yaml has
    yacs runtime-set keys that fail strict-mode reload).
    Resolution order:
      a. --orig-config <path>                          (explicit override)
      b. <ckpt_dir>/<ckpt_stem>.config.yaml            (deployment bundle)
      c. <REPO_ROOT>/configs/**/<run_name>.yaml        (repo auto-discovery)
2.  Read `cfg.dataset.task_type` to branch the rest of the flow.
3.  For classification only: resolve the threshold:
      - numeric  → use as-is
      - 'optimal-f1'  → read 'best_f1_threshold' from
                        <ckpt_dir>/<ckpt_stem>.json (manifest written by
                        retrain_on_trainval.py).
4.  Read input table (CSV or Parquet; local path or s3:// URI) with pandas
    (s3:// handled via s3fs, like the trans_learn loaders). Per row: parse
    SMILES with RDKit; invalid rows get NaN + a reason string and are skipped.
5.  Featurise valid SMILES with the SAME atom / bond featurisation as
    torch_geometric.datasets.MoleculeNet (copied inline below so this script
    is independent of the PyG version's utils namespace), then apply the
    same pre-transform chain master_loader runs for PyG-MoleculeNet:
      compute_posenc_stats(RWSE) → compute_shortest_paths → add_rings →
      line_graph → typecast_x_and_edge_attr(float).
6.  Build the model via network_dict[cfg.model.type], load checkpoint
    state_dict. `dim_out = 1` for both task types currently supported.
7.  Batched inference via PyG DataLoader.
      classification_binary  →  sigmoid(logit)         → class-1 probability
      regression             →  raw model output       → predicted target
8.  Merge predictions into original row order; for classification, apply the
    threshold to derive `y_pred_label`; write output CSV.

Output CSV
----------
All input columns preserved + appended columns (schema depends on task type):

    classification_binary:
        y_pred_score  (float)  class-1 probability; NaN for invalid SMILES
        y_pred_label  (int)    0/1 at the chosen threshold; NaN for invalid
        remarks       (str)    empty for successful rows; reason for invalid

    regression:
        y_pred        (float)  predicted target value; NaN for invalid SMILES
        remarks       (str)    empty for successful rows; reason for invalid

Evaluation (optional, --label-col)
----------------------------------
When the input CSV carries ground-truth labels, pass `--label-col <col>` to
also compute performance metrics and the same plot set as analyze_run.py
(shared `_eval_plots` module): ROC / PR / confusion @ optimal-F1 / score
histogram for classification; scatter / residuals / residual histogram for
regression. Only rows with BOTH a valid prediction and a non-null label are
scored. A `summary.json` (+ PNGs, unless `--no-plots`) is written to
`--plot-dir` (default: `<output_csv_dir>/<output_stem>_eval/`). Without
`--label-col`, prediction behaves exactly as before.

Descriptors (line_graph_with_desc models)
-----------------------------------------
If the bundled config's ``gnn.head`` is ``line_graph_with_desc``, the input CSV
must also contain the descriptor columns used in training. predict.py reads
``descriptor_columns`` + ``desc_stats`` (train-split mean/std) from the bundle
manifest (final_model.json), validates the CSV has those columns, **reorders
them to the training order** (by name — so the CSV's column order doesn't
matter), and applies the same z-score. Rows with a missing / non-finite
descriptor value get a ``remarks`` note and are skipped (NaN predictions).
Descriptors are NOT computed from SMILES — supply them as columns. For
SMILES-only (``line_graph``) models nothing changes.

Scope
-----
- Cuda-only (no --device flag — fails fast if CUDA is unavailable).
- Binary classification and single-target regression are supported.
  Multi-target regression / multi-label classification are not.
- Deployment bundle = 3 sibling files in one folder:
      final_model.ckpt
      final_model.config.yaml
      final_model.json
  Copy that trio together when moving a model to another server.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.register import network_dict
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Make the sibling _eval_plots module importable regardless of invocation/cwd.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import graphgps  # noqa: E402, F401 — registers custom modules in the GraphGym registry

from graphgps.transform.posenc_stats import compute_posenc_stats  # noqa: E402
from graphgps.transform.transforms import (  # noqa: E402
    add_rings,
    compute_shortest_paths,
    line_graph,
    typecast_x_and_edge_attr,
)

from _eval_plots import (  # noqa: E402
    analyze_classification_binary,
    analyze_regression,
)

# Atom / bond feature maps — copied verbatim from
# torch_geometric.datasets.MoleculeNet (PyG 2.0.4) so the inline featurisation
# is byte-identical to what the model saw during training.
_X_MAP = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}
_E_MAP = {
    'bond_type': [
        'misc', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC',
    ],
    'stereo': [
        'STEREONONE', 'STEREOZ', 'STEREOE',
        'STEREOCIS', 'STEREOTRANS', 'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


def _smiles_to_data(smiles: str) -> Data:
    """Replicate torch_geometric.datasets.MoleculeNet's per-SMILES featurisation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit MolFromSmiles returned None")

    xs = []
    for atom in mol.GetAtoms():
        xs.append([
            _X_MAP['atomic_num'].index(atom.GetAtomicNum()),
            _X_MAP['chirality'].index(str(atom.GetChiralTag())),
            _X_MAP['degree'].index(atom.GetTotalDegree()),
            _X_MAP['formal_charge'].index(atom.GetFormalCharge()),
            _X_MAP['num_hs'].index(atom.GetTotalNumHs()),
            _X_MAP['num_radical_electrons'].index(
                atom.GetNumRadicalElectrons()),
            _X_MAP['hybridization'].index(str(atom.GetHybridization())),
            _X_MAP['is_aromatic'].index(atom.GetIsAromatic()),
            _X_MAP['is_in_ring'].index(atom.IsInRing()),
        ])
    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = [
            _E_MAP['bond_type'].index(str(bond.GetBondType())),
            _E_MAP['stereo'].index(str(bond.GetStereo())),
            _E_MAP['is_conjugated'].index(bond.GetIsConjugated()),
        ]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    # Sort edges by (src, dst) — matches MoleculeNet's output order.
    if edge_index.numel() > 0:
        n = max(mol.GetNumAtoms(), 1)
        perm = (edge_index[0] * n + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def _resolve_config(ckpt_path: Path, orig_config: Path | None) -> Path:
    """Return the path to a pristine, yacs-reloadable YAML config."""
    if orig_config is not None:
        p = orig_config.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--orig-config not found: {p}")
        return p

    bundled = ckpt_path.parent / f'{ckpt_path.stem}.config.yaml'
    if bundled.is_file():
        return bundled

    run_name = ckpt_path.parent.name
    candidates = list((REPO_ROOT / 'configs').rglob(f'{run_name}.yaml'))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-discover the pristine config. Looked for:\n"
            f"  {bundled}\n"
            f"  {REPO_ROOT}/configs/**/{run_name}.yaml\n"
            f"Pass --orig-config <path> explicitly."
        )
    raise RuntimeError(
        f"Multiple configs match '{run_name}.yaml': {candidates}. "
        f"Pass --orig-config to disambiguate."
    )


def _resolve_threshold(ckpt_path: Path, threshold_arg: str) -> float:
    """Convert the --threshold CLI value into a float."""
    if threshold_arg == 'optimal-f1':
        manifest = ckpt_path.parent / f'{ckpt_path.stem}.json'
        if not manifest.is_file():
            raise FileNotFoundError(
                f"--threshold optimal-f1 needs the retrain manifest at "
                f"{manifest}; not found. Either pass a numeric --threshold "
                f"or run scripts/retrain_on_trainval.py to produce the manifest."
            )
        with open(manifest) as fh:
            data = json.load(fh)
        if 'best_f1_threshold' not in data:
            raise KeyError(
                f"'best_f1_threshold' missing from {manifest}. Re-run "
                f"scripts/retrain_on_trainval.py (it now records this) or "
                f"pass a numeric --threshold."
            )
        return float(data['best_f1_threshold'])
    try:
        return float(threshold_arg)
    except ValueError:
        raise ValueError(
            f"--threshold must be a float or 'optimal-f1', got: {threshold_arg!r}"
        )


def _load_desc_spec(ckpt_path: Path):
    """Read descriptor_columns + desc_stats (mean/std) from the bundle manifest.

    Required when the model's head is ``line_graph_with_desc`` — the manifest
    (written by retrain_on_trainval.py) makes prediction self-contained: no
    dataset / desc_stats.json needed at inference. Returns (columns, mean, std).
    """
    manifest = ckpt_path.parent / f'{ckpt_path.stem}.json'
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Model uses descriptors (gnn.head=line_graph_with_desc) but the "
            f"manifest {manifest} is missing — it must carry 'descriptor_columns'"
            f" + 'desc_stats'. Produce it with scripts/retrain_on_trainval.py."
        )
    with open(manifest) as fh:
        m = json.load(fh)
    cols = m.get('descriptor_columns')
    stats = m.get('desc_stats') or {}
    if not cols or 'mean' not in stats or 'std' not in stats:
        raise KeyError(
            f"{manifest} lacks 'descriptor_columns' / 'desc_stats.mean/std'. "
            f"Re-run scripts/retrain_on_trainval.py (it records them for "
            f"line_graph_with_desc models)."
        )
    mean = np.asarray(stats['mean'], dtype=float)
    std = np.asarray(stats['std'], dtype=float)
    if not (len(cols) == len(mean) == len(std)):
        raise ValueError(
            f"length mismatch in {manifest}: descriptor_columns={len(cols)}, "
            f"mean={len(mean)}, std={len(std)}."
        )
    return cols, mean, std


def _read_input_table(path: str) -> pd.DataFrame:
    """Read the input table from a local path or an ``s3://`` URI.

    Format is dispatched by extension: ``.parquet`` -> ``read_parquet``,
    otherwise ``read_csv``. pandas handles ``s3://`` transparently via s3fs
    (same mechanism the trans_learn loaders use), so no extra args/creds are
    needed beyond the AWS credentials already configured in the environment.
    """
    p = str(path)
    if p.lower().endswith('.parquet'):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _bootstrap_cfg(orig_config: Path) -> None:
    """Initialise GraphGym's global cfg from a pristine YAML."""
    set_cfg(cfg)
    cfg.merge_from_file(str(orig_config))
    cfg.device = 'cuda:0'

    # Mirror master_loader.py: turn `times_func` strings into kernel `times`.
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            if hasattr(pecfg, 'kernel') and pecfg.kernel.times_func:
                pecfg.kernel.times = list(eval(pecfg.kernel.times_func))


def _apply_pretransforms(data: Data) -> Data:
    """Apply the same pre-transform chain master_loader runs for PyG-MoleculeNet."""
    pe_enabled = [
        key.split('_', 1)[1]
        for key, pecfg in cfg.items()
        if key.startswith('posenc_')
        and pecfg.enable
        and getattr(pecfg, 'precompute', True)
    ]
    if pe_enabled:
        data = compute_posenc_stats(
            data, pe_types=pe_enabled, is_undirected=True, cfg=cfg
        )
    if getattr(cfg.dataset, 'spd', False):
        data = compute_shortest_paths(data, config=cfg.dataset)
    if getattr(cfg.dataset, 'rings', False):
        data = add_rings(data, config=cfg.dataset)
    if cfg.model.type in ('DGTModel', 'DGTModel3D', 'NodeGTModel', 'EdgeGTModel'):
        data = line_graph(data)
    # Same runtime transform as preformat_MoleculeNet.
    data = typecast_x_and_edge_attr(data, type_str='float')
    # Inference has no labels, but the GraphGym model API expects batch.y.
    data.y = torch.zeros(1, dtype=torch.long)
    return data


def _build_model(ckpt_path: Path) -> torch.nn.Module:
    """Construct the DGT model and load checkpoint weights."""
    dim_in = cfg.share.dim_in
    task_type = cfg.dataset.task_type
    # Both supported task types use a single output dim:
    #   classification_binary → 1 logit (sigmoid → probability)
    #   regression            → 1 predicted target value
    if task_type in ('classification_binary', 'regression'):
        dim_out = 1
    else:
        raise NotImplementedError(
            f"task_type={task_type!r} is not supported by predict.py. "
            f"Currently: 'classification_binary' and 'regression'."
        )

    model = network_dict[cfg.model.type](dim_in=dim_in, dim_out=dim_out)
    device = torch.device(cfg.device)
    model.to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or 'model_state' not in state:
        raise RuntimeError(
            f"Unexpected checkpoint format at {ckpt_path}: expected dict with "
            f"'model_state' key (GraphGym's save_ckpt convention)."
        )
    model.load_state_dict(state['model_state'])
    model.eval()
    return model


@torch.no_grad()
def _run_inference(model, data_list, batch_size, task_type):
    """Return a 1-D numpy array of per-sample predictions.

    classification_binary  →  class-1 probability (sigmoid applied)
    regression             →  raw model output
    """
    device = torch.device(cfg.device)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    preds = []
    for batch in loader:
        batch.split = 'test'
        batch.to(device)
        with torch.autocast(device_type='cuda'):
            pred, _ = model(batch)
        pred = pred.float().cpu()
        if task_type == 'classification_binary':
            preds.append(torch.sigmoid(pred).numpy().reshape(-1))
        else:  # regression
            preds.append(pred.numpy().reshape(-1))
    return np.concatenate(preds, axis=0) if preds else np.array([])


def _evaluate(df, task_type, label_col, plot_dir: Path, make_plots: bool):
    """Compute performance metrics (+ optional plots) against ground truth.

    Uses the prediction columns already added to `df` and the ground-truth
    `label_col`. Evaluates only rows with BOTH a valid prediction and a
    non-null label. Writes summary.json (+ plots) into `plot_dir`; returns the
    summary dict, or None if there are no evaluable rows.
    """
    if label_col not in df.columns:
        raise KeyError(
            f"--label-col '{label_col}' not in input CSV. "
            f"Available: {list(df.columns)}"
        )
    y_true_all = pd.to_numeric(df[label_col], errors='coerce').to_numpy()
    pred_col = 'y_pred_score' if task_type == 'classification_binary' else 'y_pred'
    y_pred_all = df[pred_col].to_numpy(dtype=float)

    mask = ~np.isnan(y_true_all) & ~np.isnan(y_pred_all)
    n_eval = int(mask.sum())
    if n_eval == 0:
        print("Warning: no rows with both a valid prediction and a non-null "
              "label; skipping metrics.")
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    if task_type == 'classification_binary':
        summary = analyze_classification_binary(
            y_true_all[mask], y_pred_all[mask], plot_dir, make_plots=make_plots)
    else:  # regression
        summary = analyze_regression(
            y_true_all[mask], y_pred_all[mask], plot_dir, make_plots=make_plots)

    summary['label_col'] = label_col
    summary['n_evaluated'] = n_eval
    summary['n_skipped'] = int(len(df) - n_eval)
    with open(plot_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run DGT inference on new SMILES. "
                    "Supports binary-classification and single-target "
                    "regression checkpoints; the task type is read from "
                    "the bundled config.",
    )
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to a trained .ckpt file.")
    parser.add_argument("--smiles-csv", type=str, required=True,
                        help="Input table — local path OR s3:// URI; .csv or "
                             ".parquet (by extension). Must contain the SMILES "
                             "column (and, for line_graph_with_desc models, the "
                             "descriptor columns).")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output CSV path.")
    parser.add_argument("--orig-config", type=Path, default=None,
                        help="Pristine YAML config. If omitted, looks for "
                             "<ckpt_dir>/<ckpt_stem>.config.yaml then "
                             "configs/**/<run_name>.yaml.")
    parser.add_argument("--threshold", type=str, default='0.5',
                        help="Classification-only. Decision threshold for "
                             "y_pred_label: either a float in (0,1) or "
                             "'optimal-f1' to look it up from "
                             "<ckpt_dir>/<ckpt_stem>.json. Default: 0.5. "
                             "Ignored for regression task type.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size. Default: 64.")
    parser.add_argument("--smiles-col", type=str, default='smiles',
                        help="Input CSV column holding SMILES. Default: 'smiles'.")
    parser.add_argument("--label-col", type=str, default=None,
                        help="Optional. Input CSV column holding ground-truth "
                             "labels/targets. When given, compute performance "
                             "metrics (+ plots, same as analyze_run.py) on rows "
                             "with both a valid prediction and a non-null label.")
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="Where to write the eval summary.json + plots when "
                             "--label-col is set. Default: <output_csv_dir>/"
                             "<output_stem>_eval/.")
    parser.add_argument("--no-plots", action="store_true",
                        help="With --label-col: compute metrics + write "
                             "summary.json only, skip the PNG plots.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script is cuda-only for now; "
            "rerun on a machine with a GPU."
        )

    ckpt_path = args.ckpt.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--ckpt not found: {ckpt_path}")

    orig_config = _resolve_config(ckpt_path, args.orig_config)
    print(f"Using config: {orig_config}")
    _bootstrap_cfg(orig_config)

    task_type = cfg.dataset.task_type
    print(f"Task type: {task_type}")
    if task_type == 'classification_binary':
        threshold = _resolve_threshold(ckpt_path, args.threshold)
        print(f"Using threshold: {threshold:.4f}")
    elif task_type == 'regression':
        if args.threshold != '0.5':
            print(f"Note: --threshold={args.threshold!r} ignored for "
                  "regression task.")
        threshold = None
    else:
        raise NotImplementedError(
            f"task_type={task_type!r} is not supported by predict.py. "
            f"Currently: 'classification_binary' and 'regression'."
        )

    df = _read_input_table(args.smiles_csv)
    print(f"Read {len(df)} rows from {args.smiles_csv}")
    if args.smiles_col not in df.columns:
        raise KeyError(
            f"Column '{args.smiles_col}' not in input CSV. "
            f"Available: {list(df.columns)}"
        )

    # Descriptor channel (line_graph_with_desc models only): the input CSV must
    # carry the same descriptor columns used in training. Validate by name,
    # reorder to training order, and apply the persisted train-split z-score.
    uses_desc = cfg.gnn.head == 'line_graph_with_desc'
    desc_std_mat = None
    if uses_desc:
        desc_cols, desc_mean, desc_std = _load_desc_spec(ckpt_path)
        missing = [c for c in desc_cols if c not in df.columns]
        if missing:
            shown = missing[:10] + (['...'] if len(missing) > 10 else [])
            raise KeyError(
                f"Input CSV is missing {len(missing)} descriptor column(s) this "
                f"model needs: {shown}. A line_graph_with_desc model requires the "
                f"same descriptors as training (see final_model.json "
                f"'descriptor_columns')."
            )
        desc_raw = (df[desc_cols].apply(pd.to_numeric, errors='coerce')
                    .to_numpy(dtype=float))           # reordered to training order
        desc_std_mat = (desc_raw - desc_mean) / desc_std
        print(f"Descriptor channel: {len(desc_cols)} columns (standardised).")

    valid_indices, valid_data, remarks = [], [], [''] * len(df)
    for i, smi in enumerate(df[args.smiles_col].astype(str).tolist()):
        try:
            data = _smiles_to_data(smi)
            data = _apply_pretransforms(data)
        except Exception as e:
            remarks[i] = f"invalid SMILES: {e}"
            continue
        if uses_desc:
            drow = desc_std_mat[i]
            if not np.isfinite(drow).all():
                remarks[i] = "missing/non-finite descriptor value(s)"
                continue
            data.desc = torch.tensor(drow, dtype=torch.float).view(1, -1)
        valid_data.append(data)
        valid_indices.append(i)
    print(f"Featurised {len(valid_indices)} / {len(df)} rows; "
          f"{len(df) - len(valid_indices)} invalid.")

    model = _build_model(ckpt_path)
    preds = _run_inference(model, valid_data, args.batch_size, task_type)
    assert len(preds) == len(valid_indices), \
        f"Prediction count {len(preds)} != valid row count {len(valid_indices)}"

    if task_type == 'classification_binary':
        y_score = np.full(len(df), np.nan, dtype=float)
        y_label = np.full(len(df), np.nan, dtype=float)
        for idx, s in zip(valid_indices, preds):
            y_score[idx] = float(s)
            y_label[idx] = 1.0 if s >= threshold else 0.0
        df['y_pred_score'] = y_score
        df['y_pred_label'] = y_label
    else:  # regression
        y_pred = np.full(len(df), np.nan, dtype=float)
        for idx, p in zip(valid_indices, preds):
            y_pred[idx] = float(p)
        df['y_pred'] = y_pred
    df['remarks'] = remarks

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions: {args.output_csv}")

    # Optional evaluation: metrics (+ plots) when ground-truth labels are given.
    if args.label_col is not None:
        plot_dir = (args.plot_dir.resolve() if args.plot_dir is not None
                    else args.output_csv.resolve().parent
                    / f"{args.output_csv.stem}_eval")
        summary = _evaluate(df, task_type, args.label_col, plot_dir,
                            make_plots=not args.no_plots)
        if summary is not None:
            where = ("summary only (no plots)" if args.no_plots
                     else "plots + summary")
            print(f"Evaluation complete ({where}): {plot_dir}")
            print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
