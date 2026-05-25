"""Inference on new SMILES with a trained DGT classifier.

CLI
---
    python scripts/predict.py \\
        --ckpt        <path/to/final_model.ckpt>     (required)
        --smiles-csv  <input.csv>                    (required)
        --output-csv  <output.csv>                   (required)
        [--orig-config <path>]                       (default: auto-discover)
        [--threshold  0.5 | optimal-f1]              (default: 0.5)
        [--batch-size 64]                            (default: 64)
        [--smiles-col smiles]                        (default: 'smiles')

Inference flow
--------------
1.  Resolve the *pristine* YAML config (the dumped <run_dir>/config.yaml has
    yacs runtime-set keys that fail strict-mode reload).
    Resolution order:
      a. --orig-config <path>                          (explicit override)
      b. <ckpt_dir>/<ckpt_stem>.config.yaml            (deployment bundle)
      c. <REPO_ROOT>/configs/**/<run_name>.yaml        (repo auto-discovery)
2.  Resolve the threshold:
      - numeric  → use as-is
      - 'optimal-f1'  → read 'best_f1_threshold' from
                        <ckpt_dir>/<ckpt_stem>.json (manifest written by
                        retrain_on_trainval.py).
3.  Read input CSV (pandas). Per row: parse SMILES with RDKit; invalid rows
    get NaN + a reason string and are skipped during inference.
4.  Featurise valid SMILES with the SAME atom / bond featurisation as
    torch_geometric.datasets.MoleculeNet (copied inline below so this script
    is independent of the PyG version's utils namespace), then apply the
    same pre-transform chain master_loader runs for PyG-MoleculeNet:
      compute_posenc_stats(RWSE) → compute_shortest_paths → add_rings →
      line_graph → typecast_x_and_edge_attr(float).
5.  Build the model via network_dict[cfg.model.type], load checkpoint
    state_dict.
6.  Batched inference via PyG DataLoader. Logits are sigmoid'd (binary
    classification) for class-1 probability.
7.  Merge predictions into original row order; apply threshold for
    y_pred_label; write output CSV.

Output CSV
----------
All input columns preserved + 3 appended columns:
    y_pred_score  (float)  class-1 probability; NaN for invalid SMILES
    y_pred_label  (int)    0/1 at the chosen threshold; NaN for invalid
    remarks       (str)    empty for successful rows; reason for invalid

Scope
-----
- Cuda-only (no --device flag — fails fast if CUDA is unavailable).
- Binary classification only. Regression is a follow-up.
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

import graphgps  # noqa: E402, F401 — registers custom modules in the GraphGym registry

from graphgps.transform.posenc_stats import compute_posenc_stats  # noqa: E402
from graphgps.transform.transforms import (  # noqa: E402
    add_rings,
    compute_shortest_paths,
    line_graph,
    typecast_x_and_edge_attr,
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
        'UNSPECIFIED', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC',
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
    # Mirror main.py:create_model — binary classification collapses dim_out 2 → 1.
    dim_out = 2
    if (
        'classification' in cfg.dataset.task_type
        and 'multilabel' not in cfg.dataset.task_type
        and dim_out == 2
    ):
        dim_out = 1

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
def _run_inference(model, data_list, batch_size):
    """Return a 1-D numpy array of class-1 probabilities, one per Data."""
    device = torch.device(cfg.device)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    scores = []
    for batch in loader:
        batch.split = 'test'
        batch.to(device)
        with torch.autocast(device_type='cuda'):
            pred, _ = model(batch)
        scores.append(torch.sigmoid(pred.float()).cpu().numpy().reshape(-1))
    return np.concatenate(scores, axis=0) if scores else np.array([])


def main():
    parser = argparse.ArgumentParser(
        description="Run DGT inference on new SMILES "
                    "(binary classification only).",
    )
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Path to a trained .ckpt file.")
    parser.add_argument("--smiles-csv", type=Path, required=True,
                        help="Input CSV; must contain the SMILES column.")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output CSV path.")
    parser.add_argument("--orig-config", type=Path, default=None,
                        help="Pristine YAML config. If omitted, looks for "
                             "<ckpt_dir>/<ckpt_stem>.config.yaml then "
                             "configs/**/<run_name>.yaml.")
    parser.add_argument("--threshold", type=str, default='0.5',
                        help="Decision threshold for y_pred_label. Either a "
                             "float in (0,1) or 'optimal-f1' to look it up "
                             "from <ckpt_dir>/<ckpt_stem>.json. Default: 0.5.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size. Default: 64.")
    parser.add_argument("--smiles-col", type=str, default='smiles',
                        help="Input CSV column holding SMILES. Default: 'smiles'.")
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

    threshold = _resolve_threshold(ckpt_path, args.threshold)
    print(f"Using threshold: {threshold:.4f}")

    df = pd.read_csv(args.smiles_csv)
    if args.smiles_col not in df.columns:
        raise KeyError(
            f"Column '{args.smiles_col}' not in input CSV. "
            f"Available: {list(df.columns)}"
        )

    valid_indices, valid_data, remarks = [], [], [''] * len(df)
    for i, smi in enumerate(df[args.smiles_col].astype(str).tolist()):
        try:
            data = _smiles_to_data(smi)
            data = _apply_pretransforms(data)
            valid_data.append(data)
            valid_indices.append(i)
        except Exception as e:
            remarks[i] = f"invalid SMILES: {e}"
    print(f"Featurised {len(valid_indices)} / {len(df)} rows; "
          f"{len(df) - len(valid_indices)} invalid.")

    model = _build_model(ckpt_path)
    scores = _run_inference(model, valid_data, args.batch_size)
    assert len(scores) == len(valid_indices), \
        f"Score count {len(scores)} != valid row count {len(valid_indices)}"

    y_score = np.full(len(df), np.nan, dtype=float)
    y_label = np.full(len(df), np.nan, dtype=float)
    for idx, s in zip(valid_indices, scores):
        y_score[idx] = float(s)
        y_label[idx] = 1.0 if s >= threshold else 0.0

    df['y_pred_score'] = y_score
    df['y_pred_label'] = y_label
    df['remarks'] = remarks

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions: {args.output_csv}")


if __name__ == '__main__':
    main()
