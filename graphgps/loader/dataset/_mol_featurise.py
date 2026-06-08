"""Shared molecular featurisation for the biodegradability PyG loaders.

Single source of truth for [biodeg_gwu.py](biodeg_gwu.py) and
[biodeg.py](biodeg.py): both import ``smiles_to_xy`` from here so the two
datasets are featurised identically. ``X_MAP`` / ``E_MAP`` / ``smiles_to_xy``
are copied verbatim from ``torch_geometric.datasets.MoleculeNet`` (PyG 2.0.4),
so the atom/bond encoding matches what BBBP / FreeSolv saw at training time and
what the DGT encoders expect (9 categorical atom features, 3 bond features).

NOTE: ``scripts/predict.py`` deliberately keeps its OWN copy of this code — it
is a self-contained deployment artifact, meant to be copied to another server
without the ``graphgps`` package, so it must not import from here. If you ever
change the featurisation, change it in BOTH places and re-run
``pytest -rP tests/test_dataset.py`` to confirm the loaders still build valid
caches.
"""
import torch
from rdkit import Chem

X_MAP = {
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
E_MAP = {
    'bond_type': [
        'misc', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC',
    ],
    'stereo': [
        'STEREONONE', 'STEREOZ', 'STEREOE',
        'STEREOCIS', 'STEREOTRANS', 'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


def smiles_to_xy(smiles):
    """SMILES → (x, edge_index, edge_attr) or None if RDKit can't parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    xs = []
    for atom in mol.GetAtoms():
        xs.append([
            X_MAP['atomic_num'].index(atom.GetAtomicNum()),
            X_MAP['chirality'].index(str(atom.GetChiralTag())),
            X_MAP['degree'].index(atom.GetTotalDegree()),
            X_MAP['formal_charge'].index(atom.GetFormalCharge()),
            X_MAP['num_hs'].index(atom.GetTotalNumHs()),
            X_MAP['num_radical_electrons'].index(
                atom.GetNumRadicalElectrons()),
            X_MAP['hybridization'].index(str(atom.GetHybridization())),
            X_MAP['is_aromatic'].index(atom.GetIsAromatic()),
            X_MAP['is_in_ring'].index(atom.IsInRing()),
        ])
    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = [
            E_MAP['bond_type'].index(str(bond.GetBondType())),
            E_MAP['stereo'].index(str(bond.GetStereo())),
            E_MAP['is_conjugated'].index(bond.GetIsConjugated()),
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
