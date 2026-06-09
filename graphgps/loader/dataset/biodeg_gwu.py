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
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from graphgps.loader.dataset._mol_featurise import smiles_to_xy
from graphgps.loader.dataset._desc_select import (
    select_descriptor_columns,
    selection_tag,
)


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

    def __init__(self, root, standardize_desc=False, desc_include=None,
                 desc_exclude=None, desc_columns=None, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = 'biodeg_gwu'
        # standardize_desc: z-score descriptors with train-split stats.
        # desc_include/exclude/columns: subset which descriptors are used.
        # Both change desc content -> a SEPARATE processed cache keyed by the
        # resolved selection (resolved before super().__init__, which reads
        # processed_file_names to decide whether process() must run).
        self.standardize_desc = standardize_desc
        self.desc_include = list(desc_include or [])
        self.desc_exclude = list(desc_exclude or [])
        self.desc_columns = list(desc_columns or [])
        self._resolve_desc_selection(root)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.parquet', 'test.parquet', 'manifest.json']

    @property
    def processed_file_names(self):
        # Separate cache per (standardisation, descriptor-selection) so the
        # baseline raw-desc cache and each subset never collide.
        if not self.standardize_desc and self._selected_columns is None:
            return 'data.pt'
        prefix = 'data_stdesc' if self.standardize_desc else 'data_rawdesc'
        return f'{prefix}{self._cache_suffix}.pt'

    def _resolve_desc_selection(self, root):
        """Resolve the descriptor-column subset + cache suffix from the manifest.

        Reads ``root/raw/manifest.json`` (if present) for the full column list,
        applies include/exclude/columns, and derives a short hash suffix for the
        cache filename. Leaves selection unset (full set) when no spec is given
        or the manifest is absent (download() then raises the prepare hint).
        """
        self._selected_columns = None
        self._cache_suffix = ''
        if not (self.desc_include or self.desc_exclude or self.desc_columns):
            return
        manifest_path = osp.join(root, 'raw', 'manifest.json')
        if not osp.isfile(manifest_path):
            return
        with open(manifest_path) as fh:
            all_cols = json.load(fh)['descriptor_columns']
        self._selected_columns = select_descriptor_columns(
            all_cols, self.desc_include, self.desc_exclude, self.desc_columns)
        self._cache_suffix = '_' + selection_tag(self._selected_columns)

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
        # Optional descriptor-column selection (Phase 2 study); effective
        # desc_dim is the SELECTED count, not the manifest's full count.
        if self._selected_columns is not None:
            descriptor_cols = self._selected_columns
            logging.info(f"{self.name}: descriptor selection -> "
                         f"{len(descriptor_cols)} cols (cache suffix "
                         f"'{self._cache_suffix}').")
        desc_dim = len(descriptor_cols)

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

        # Optional descriptor standardisation (Phase 2, Option A): z-score with
        # TRAIN-split stats only (no val/test leakage), then persist stats +
        # column names so val/test/predict reuse identical normalisation.
        desc_mean = desc_std = None
        if self.standardize_desc:
            train_desc = df_train[descriptor_cols].astype(float).to_numpy()
            is_train = np.array(
                [i not in val_idx_set for i in range(n_train_full)]
            )
            train_only = train_desc[is_train]
            desc_mean = train_only.mean(axis=0)
            desc_std = train_only.std(axis=0)
            # Constant columns (std~0): set std=1 so (x-mean)=0 maps to 0.
            desc_std = np.where(desc_std < 1e-8, 1.0, desc_std)
            stats_path = osp.join(self.processed_dir, 'desc_stats.json')
            with open(stats_path, 'w') as fh:
                json.dump({
                    'descriptor_columns': list(descriptor_cols),
                    'desc_dim': desc_dim,
                    'mean': desc_mean.tolist(),
                    'std': desc_std.tolist(),
                    'standardize': True,
                    'split_seed': self.SPLIT_SEED,
                    'val_fraction': self.VAL_FRACTION,
                }, fh, indent=2)
            logging.info(f"{self.name}: standardised descriptors (train-split "
                         f"stats); wrote {stats_path}")

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

                feats = smiles_to_xy(smiles)
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
                desc_vec = desc_array[row_idx]
                if desc_mean is not None:
                    desc_vec = (desc_vec - desc_mean) / desc_std
                desc = torch.tensor(desc_vec, dtype=torch.float).view(1, -1)
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
