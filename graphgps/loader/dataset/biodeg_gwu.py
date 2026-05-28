"""biodeg_gwu PyG dataset — biodegradability (GWU batch 2).

Reads the local snapshot written by ``scripts/prepare_data.py``:
    <root>/raw/train.parquet
    <root>/raw/test.parquet
    <root>/raw/manifest.json

and emits PyG ``Data`` objects featurised the same way as
``torch_geometric.datasets.MoleculeNet`` so the rest of the DGT pipeline
(BBBP-style featurisation + encoders + DGT layers) sees byte-identical
input shapes.

Each Data carries:
    x          [N, 9]            atom features (long; cast to float in
                                 master_loader via typecast_x_and_edge_attr)
    edge_index [2, 2M]           bidirectional atom-graph edges
    edge_attr  [2M, 3]           bond features (long; cast to float in
                                 master_loader)
    y          [1, 1]            binary label (float)
    desc       [1, desc_dim]     molecular descriptor vector (float;
                                 standardisation deferred to Phase 2)
    smiles     str               original SMILES (for analysis/plots)
    split      str               'train' / 'val' / 'test'

Split convention: the prepared train parquet is partitioned 90/10 into
train/val with a fixed seed for reproducibility; the prepared test
parquet maps 1:1 to the test split. The per-Data ``split`` attribute is
consumed by ``preformat_BiodegGwu`` in master_loader.py to build the
``train_graph_index`` / ``val_graph_index`` / ``test_graph_index`` lists
expected by ``split_mode: standard``.

No trans_learn import — the loader is self-contained and reads only
local files written by the prepare script.
"""
import json
import logging
import os.path as osp

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

# Atom / bond feature maps — copied verbatim from
# torch_geometric.datasets.MoleculeNet (PyG 2.0.4) so featurisation matches
# what BBBP / FreeSolv saw at training time, and what scripts/predict.py
# emits at inference time.
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


def _smiles_to_xy(smiles):
    """SMILES → (x, edge_index, edge_attr) or None if RDKit can't parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

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

    return x, edge_index, edge_attr


class BiodegGwu(InMemoryDataset):
    """PyG InMemoryDataset for the biodeg_gwu dataset.

    Args:
        root: directory containing ``raw/{train,test}.parquet`` +
            ``raw/manifest.json`` (the output of ``scripts/prepare_data.py
            --dataset biodeg_gwu``).
        transform, pre_transform, pre_filter: standard PyG hooks.
    """

    # Fraction of the prepared train parquet held out as validation.
    VAL_FRACTION = 0.10
    # Fixed seed for the 90/10 train/val split — change only if you
    # deliberately want a different held-out set.
    SPLIT_SEED = 42

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = 'biodeg_gwu'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.parquet', 'test.parquet', 'manifest.json']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # No automatic download — this dataset is fetched offline by
        # scripts/prepare_data.py. Surface a clear message if raw files
        # are missing so the user knows what to do next.
        missing = [f for f in self.raw_file_names
                   if not osp.isfile(osp.join(self.raw_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"Missing raw files in {self.raw_dir}: {missing}\n"
                f"Run the prepare script first:\n"
                f"  python scripts/prepare_data.py "
                f"--dataset biodeg_gwu --trans-learn-path <path>\n"
                f"(See documents/overview.md Phase 1 for details.)"
            )

    def process(self):
        # 1. Load the prepare-script outputs.
        with open(osp.join(self.raw_dir, 'manifest.json')) as fh:
            manifest = json.load(fh)
        smiles_col = manifest['smiles_column']
        target_col = manifest['target_column']
        descriptor_cols = manifest['descriptor_columns']
        desc_dim = manifest['desc_dim']

        df_train = pd.read_parquet(osp.join(self.raw_dir, 'train.parquet'))
        df_test = pd.read_parquet(osp.join(self.raw_dir, 'test.parquet'))

        # ─────────────────────────────────────────────────────────────────
        # SPLIT CONVENTION
        #   train.parquet  →  90% train  +  10% val   (random, fixed seed)
        #   test.parquet   →  100% test  (untouched — never mixed into the
        #                                 val carve-out)
        # The held-out test split on disk stays exactly the held-out test
        # split used here. Only the train parquet is subdivided.
        # ─────────────────────────────────────────────────────────────────
        rng = np.random.default_rng(self.SPLIT_SEED)
        n_train_full = len(df_train)
        perm = rng.permutation(n_train_full)
        n_val = int(round(n_train_full * self.VAL_FRACTION))
        val_idx_set = set(perm[:n_val].tolist())

        # 3. Featurise each row.
        # NB: pre-extracting NumPy arrays (not df.itertuples) — itertuples
        # builds namedtuples whose field names must be valid Python identifiers,
        # so columns like `COC(C)=O_fg` (SMILES-fragment functional-group flags)
        # get renamed positionally and can no longer be looked up by their
        # original name. Array indexing sidesteps this and is also faster.
        data_list = []
        dropped = {'train': 0, 'val': 0, 'test': 0}
        for split_source, df in (('train', df_train), ('test', df_test)):
            smiles_values = df[smiles_col].astype(str).to_numpy()
            target_values = df[target_col].astype(float).to_numpy()
            desc_array = df[descriptor_cols].astype(float).to_numpy()

            for row_idx in tqdm(
                range(len(df)),
                desc=f"biodeg_gwu: featurising {split_source}",
            ):
                smiles = smiles_values[row_idx]

                feats = _smiles_to_xy(smiles)
                # Decide the split tag BEFORE skipping invalid rows, so the
                # `dropped` counter is split-correct.
                if split_source == 'test':
                    split_tag = 'test'
                else:
                    split_tag = 'val' if row_idx in val_idx_set else 'train'

                if feats is None:
                    logging.warning(
                        f"biodeg_gwu: dropping unparseable SMILES "
                        f"(source={split_source}, row={row_idx}): {smiles!r}"
                    )
                    dropped[split_tag] += 1
                    continue
                x, edge_index, edge_attr = feats

                y = torch.tensor(
                    [float(target_values[row_idx])], dtype=torch.float
                ).view(1, -1)

                # ─────────────────────────────────────────────────────────
                # MOLECULAR DESCRIPTORS ENTER HERE — and nowhere else.
                # `desc` is a graph-level tensor of shape [1, desc_dim].
                # PyG's mini-batch collation stacks it to [B, desc_dim] and
                # carries it through the model untouched: encoders, DGT
                # attention layers, and pairwise tensors (SPDE / RWSE / RSE)
                # never see it. The Phase-2 `line_graph_with_desc` head is
                # the only component that reads `batch.desc` — concatenating
                # it with the pooled atom/bond embeddings at readout.
                # See documents/overview.md Phase 2 for the design rationale.
                # ─────────────────────────────────────────────────────────
                desc = torch.tensor(
                    desc_array[row_idx], dtype=torch.float
                ).view(1, -1)
                assert desc.shape == (1, desc_dim), (
                    f"desc shape {desc.shape} != (1, {desc_dim}) — "
                    f"check the manifest against the parquet columns."
                )

                data = Data(
                    x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=y, desc=desc, smiles=smiles, split=split_tag,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        # 4. Log a summary so misconfigurations show up immediately.
        n_by_split = {'train': 0, 'val': 0, 'test': 0}
        pos_by_split = {'train': 0, 'val': 0, 'test': 0}
        for d in data_list:
            n_by_split[d.split] += 1
            if int(d.y.item()) == 1:
                pos_by_split[d.split] += 1
        logging.info(
            f"biodeg_gwu: featurised {len(data_list)} molecules "
            f"(dropped {sum(dropped.values())}: {dropped})."
        )
        for s in ('train', 'val', 'test'):
            n = n_by_split[s]
            p = pos_by_split[s]
            frac = (p / n) if n else 0.0
            logging.info(
                f"  {s}: n={n}, positives={p} ({frac:.1%}), negatives={n - p}"
            )

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return f'BiodegGwu({len(self)})'
