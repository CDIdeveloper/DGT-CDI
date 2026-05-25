import os
import os.path as osp
import re

import torch
from tqdm import tqdm

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz)

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


class Chiral3DMoleculeNet(InMemoryDataset):

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        'bace': ['BACE', 'bace.csv', 'bace', 0, 2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21', -1, 3],
        'chiro': ['ChIRo', 'chiro.csv.gz', 'chiro', 0, 0],
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name.lower()
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        if self.name == 'chiro':
            return 'train.pkl', 'val.pkl', 'test.pkl'
        else:
            return f'{self.names[self.name][2]}.csv', f'{self.names[self.name][2]}.sdf'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        from rdkit import Chem
        import pickle
        import pandas as pd

        if self.name == 'chiro':
            train_dataset = pickle.load(open(self.raw_paths[0], 'rb'))
            val_dataset = pickle.load(open(self.raw_paths[1], 'rb'))
            test_dataset = pickle.load(open(self.raw_paths[2], 'rb'))
            dataset = pd.concat([train_dataset, val_dataset, test_dataset], axis=0)
            dataset.reset_index(inplace=True, drop=True)

            smis = []
            data_list = []
            for i in range(len(dataset)):
                smiles = dataset.loc[i, 'ID']
                if smiles in smis:
                    continue
                smis.append(smiles)

                ys = dataset.loc[i, 'RS_label_binary'].item()
                ys = ys if isinstance(ys, list) else [ys]
                y = torch.tensor(ys, dtype=torch.float).view(1, -1)
                if torch.isnan(y).all():
                    continue

                if i < len(train_dataset):
                    split = 'train'
                elif i < len(train_dataset) + len(val_dataset):
                    split = 'val'
                else:
                    split = 'test'

                mol = dataset.loc[i, 'rdkit_mol_cistrans_stereo']
                Chem.AssignStereochemistry(mol, 
                    flagPossibleStereoCenters=True, force=True)
                pos = mol.GetConformer().GetPositions()
                pos = torch.tensor(pos, dtype=torch.float)

                xs = []
                chiral_es = {}
                for atom in mol.GetAtoms():
                    x = []
                    x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                    x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                    x.append(x_map['degree'].index(atom.GetTotalDegree()))
                    x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                    x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                    x.append(x_map['num_radical_electrons'].index(
                        atom.GetNumRadicalElectrons()))
                    x.append(x_map['hybridization'].index(
                        str(atom.GetHybridization())) if str(atom.GetHybridization()) in x_map['hybridization'] else len(x_map['hybridization']) - 1)
                    x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                    x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                    
                    xs.append(x)
                    
                    if atom.GetChiralTag() in [Chem.CHI_TETRAHEDRAL_CW, 
                        Chem.CHI_TETRAHEDRAL_CCW] and \
                        atom.GetChiralTag() == atom.GetChiralTag():
                        neighbors = atom.GetNeighbors()
                        neighbor_ranks = []
                        for n in neighbors:
                            rank = int(n.GetProp('_CIPRank'))
                            neighbor_ranks.append((n.GetIdx(), rank))
                        neighbor_ranks = sorted(neighbor_ranks, key=lambda x: x[1], reverse=True)
                        cip_order = [idx for idx, rank in neighbor_ranks]
                        
                        coords = pos[atom.GetIdx()]
                        neighbor_coords = pos[cip_order]
                        if len(neighbors) == 4:
                            v = neighbor_coords - coords
                        elif len(neighbors) == 3:
                            v4 = (neighbor_coords - coords).sum(dim=0, keepdims=True)
                            v4 = -v4 / torch.norm(v4, p=2, dim=1) * 0.6
                            v = torch.concat([neighbor_coords - coords, v4], dim=0)
                        else:
                            raise ValueError('Neighbors less than 3')
                        quads = [(0,1,2,3), (0,1,3,2), (0,2,3,1), (1,2,3,0)]
                        for a, b, c, d in quads:
                            vol = torch.dot(v[a], torch.cross(v[b], v[c])) / 6.0
                            if d < len(cip_order):
                                chiral_es[(atom.GetIdx(), cip_order[d])] = vol

                x = torch.tensor(xs, dtype=torch.float).view(-1, 9)

                edge_indices, edge_attrs = [], []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    e1 = []
                    e1.append(e_map['bond_type'].index(str(bond.GetBondType())))
                    e1.append(e_map['stereo'].index(str(bond.GetStereo())))
                    e1.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
                    e1.append(chiral_es.get((i, j), 0))
                    e1.append(chiral_es.get((j, i), 0))

                    e2 = []
                    e2.append(e_map['bond_type'].index(str(bond.GetBondType())))
                    e2.append(e_map['stereo'].index(str(bond.GetStereo())))
                    e2.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
                    e2.append(chiral_es.get((j, i), 0))
                    e2.append(chiral_es.get((i, j), 0))

                    edge_indices += [[i, j], [j, i]]
                    edge_attrs += [e1, e2]

                edge_index = torch.tensor(edge_indices)
                edge_index = edge_index.t().to(torch.long).view(2, -1)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 5)

                # Sort indices.
                if edge_index.numel() > 0:
                    perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

                unique_edge_indices = [edge_index for edge_index in edge_indices 
                                    if edge_index[0] < edge_index[1]]
                iso_edge_indices, iso_edge_attr = [], []
                for bond in mol.GetBonds():
                    if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        stereo = bond.GetStereo()
                        if stereo in (Chem.rdchem.BondStereo.STEREOE, 
                                    Chem.rdchem.BondStereo.STEREOZ,
                                    Chem.rdchem.BondStereo.STEREOCIS, 
                                    Chem.rdchem.BondStereo.STEREOTRANS):
                            C1_idx = bond.GetBeginAtomIdx()
                            C2_idx = bond.GetEndAtomIdx()
                            e = unique_edge_indices.index(sorted((C1_idx, C2_idx)))
                            stereo_atoms = bond.GetStereoAtoms()
                            if len(stereo_atoms) < 2:
                                continue
                            base_a, base_b = stereo_atoms[0], stereo_atoms[1]
                            is_cis_base = stereo in (Chem.rdchem.BondStereo.STEREOZ, 
                                                    Chem.rdchem.BondStereo.STEREOCIS)
                            n_C1 = [a.GetIdx() 
                                    for a in mol.GetAtomWithIdx(C1_idx).GetNeighbors() 
                                    if a.GetIdx() != C2_idx]
                            n_C2 = [a.GetIdx() 
                                    for a in mol.GetAtomWithIdx(C2_idx).GetNeighbors() 
                                    if a.GetIdx() != C1_idx]
                            if base_a not in n_C1:
                                base_a, base_b = base_b, base_a
                            for n1 in n_C1:
                                for n2 in n_C2:
                                    is_base_a = (n1 == base_a)
                                    is_base_b = (n2 == base_b)
                                    if is_base_a == is_base_b:
                                        pair_is_cis = is_cis_base
                                    else:
                                        pair_is_cis = not is_cis_base
                                    rel = 1 if pair_is_cis else 2
                                    e1 = unique_edge_indices.index(sorted((n1, C1_idx)))
                                    e2 = unique_edge_indices.index(sorted((n2, C2_idx)))
                                    iso_edge_indices += [[e1, e2], [e2, e1]]
                                    iso_edge_attr += [rel, rel]

                iso_edge_index = torch.tensor(iso_edge_indices)
                iso_edge_index = iso_edge_index.t().to(torch.long).view(2, -1)
                iso_edge_attr = torch.tensor(iso_edge_attr, dtype=torch.long)

                # Sort indices.
                if iso_edge_index.numel() > 0:
                    perm = (iso_edge_index[0] * edge_attr.size(0) // 2 + iso_edge_index[1]).argsort()
                    iso_edge_index, iso_edge_attr = iso_edge_index[:, perm], iso_edge_attr[perm]

                pos = pos[:mol.GetNumAtoms()]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                            iso_edge_index=iso_edge_index, iso_edge_attr=iso_edge_attr,
                            pos=pos, y=y, smiles=smiles, split=split)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        else:
            with open(self.raw_paths[0], 'r') as f:
                dataset = f.read().split('\n')[1:-1]
                dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.
            
            sdf_mols = {
                int(mol.GetProp('_Name').split('_')[0]): mol for mol in 
                Chem.SDMolSupplier(self.raw_paths[1], sanitize=False, removeHs=False)
                if mol.GetProp('_Name').split('_')[1] == 'succeeded'
            }

            data_list = []
            for i, line in enumerate(dataset):
                line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
                line = line.split(',')

                smiles = line[self.names[self.name][3]]
                ys = line[self.names[self.name][4]]
                ys = ys if isinstance(ys, list) else [ys]

                ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
                y = torch.tensor(ys, dtype=torch.float).view(1, -1)
                if torch.isnan(y).all():
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                sdf_mol = sdf_mols.get(i, None)
                if sdf_mol:
                    Chem.AssignStereochemistry(sdf_mol, 
                        flagPossibleStereoCenters=True, force=True)
                    pos = sdf_mol.GetConformer().GetPositions()
                    pos = torch.tensor(pos, dtype=torch.float)

                xs = []
                chiral_es = {}
                for atom in mol.GetAtoms():
                    x = []
                    x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                    x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                    x.append(x_map['degree'].index(atom.GetTotalDegree()))
                    x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                    x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                    x.append(x_map['num_radical_electrons'].index(
                        atom.GetNumRadicalElectrons()))
                    x.append(x_map['hybridization'].index(
                        str(atom.GetHybridization())))
                    x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                    x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                    
                    xs.append(x)
                    
                    if sdf_mol:
                        sdf_atom = sdf_mol.GetAtomWithIdx(atom.GetIdx())
                        if sdf_atom.GetChiralTag() in [Chem.CHI_TETRAHEDRAL_CW, 
                            Chem.CHI_TETRAHEDRAL_CCW] and \
                            atom.GetChiralTag() == sdf_atom.GetChiralTag():
                            neighbors = sdf_atom.GetNeighbors()
                            neighbor_ranks = []
                            for n in neighbors:
                                rank = int(n.GetProp('_CIPRank'))
                                neighbor_ranks.append((n.GetIdx(), rank))
                            neighbor_ranks = sorted(neighbor_ranks, key=lambda x: x[1], reverse=True)
                            cip_order = [idx for idx, rank in neighbor_ranks]
                            
                            coords = pos[sdf_atom.GetIdx()]
                            neighbor_coords = pos[cip_order]
                            if len(neighbors) == 4:
                                v = neighbor_coords - coords
                            elif len(neighbors) == 3:
                                v4 = (neighbor_coords - coords).sum(dim=0, keepdims=True)
                                v4 = -v4 / torch.norm(v4, p=2, dim=1) * 0.6
                                v = torch.concat([neighbor_coords - coords, v4], dim=0)
                            else:
                                raise ValueError('Neighbors less than 3')
                            quads = [(0,1,2,3), (0,1,3,2), (0,2,3,1), (1,2,3,0)]
                            for a, b, c, d in quads:
                                vol = torch.dot(v[a], torch.cross(v[b], v[c])) / 6.0
                                if d < len(cip_order):
                                    chiral_es[(atom.GetIdx(), cip_order[d])] = vol

                x = torch.tensor(xs, dtype=torch.float).view(-1, 9)

                edge_indices, edge_attrs = [], []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    e1 = []
                    e1.append(e_map['bond_type'].index(str(bond.GetBondType())))
                    e1.append(e_map['stereo'].index(str(bond.GetStereo())))
                    e1.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
                    e1.append(chiral_es.get((i, j), 0))
                    e1.append(chiral_es.get((j, i), 0))

                    e2 = []
                    e2.append(e_map['bond_type'].index(str(bond.GetBondType())))
                    e2.append(e_map['stereo'].index(str(bond.GetStereo())))
                    e2.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
                    e2.append(chiral_es.get((j, i), 0))
                    e2.append(chiral_es.get((i, j), 0))

                    edge_indices += [[i, j], [j, i]]
                    edge_attrs += [e1, e2]

                edge_index = torch.tensor(edge_indices)
                edge_index = edge_index.t().to(torch.long).view(2, -1)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 5)

                # Sort indices.
                if edge_index.numel() > 0:
                    perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

                unique_edge_indices = [edge_index for edge_index in edge_indices 
                                    if edge_index[0] < edge_index[1]]
                iso_edge_indices, iso_edge_attr = [], []
                for bond in mol.GetBonds():
                    if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        stereo = bond.GetStereo()
                        if stereo in (Chem.rdchem.BondStereo.STEREOE, 
                                    Chem.rdchem.BondStereo.STEREOZ,
                                    Chem.rdchem.BondStereo.STEREOCIS, 
                                    Chem.rdchem.BondStereo.STEREOTRANS):
                            C1_idx = bond.GetBeginAtomIdx()
                            C2_idx = bond.GetEndAtomIdx()
                            e = unique_edge_indices.index(sorted((C1_idx, C2_idx)))
                            stereo_atoms = bond.GetStereoAtoms()
                            if len(stereo_atoms) < 2:
                                continue
                            base_a, base_b = stereo_atoms[0], stereo_atoms[1]
                            is_cis_base = stereo in (Chem.rdchem.BondStereo.STEREOZ, 
                                                    Chem.rdchem.BondStereo.STEREOCIS)
                            n_C1 = [a.GetIdx() 
                                    for a in mol.GetAtomWithIdx(C1_idx).GetNeighbors() 
                                    if a.GetIdx() != C2_idx]
                            n_C2 = [a.GetIdx() 
                                    for a in mol.GetAtomWithIdx(C2_idx).GetNeighbors() 
                                    if a.GetIdx() != C1_idx]
                            if base_a not in n_C1:
                                base_a, base_b = base_b, base_a
                            for n1 in n_C1:
                                for n2 in n_C2:
                                    is_base_a = (n1 == base_a)
                                    is_base_b = (n2 == base_b)
                                    if is_base_a == is_base_b:
                                        pair_is_cis = is_cis_base
                                    else:
                                        pair_is_cis = not is_cis_base
                                    rel = 1 if pair_is_cis else 2
                                    e1 = unique_edge_indices.index(sorted((n1, C1_idx)))
                                    e2 = unique_edge_indices.index(sorted((n2, C2_idx)))
                                    iso_edge_indices += [[e1, e2], [e2, e1]]
                                    iso_edge_attr += [rel, rel]

                iso_edge_index = torch.tensor(iso_edge_indices)
                iso_edge_index = iso_edge_index.t().to(torch.long).view(2, -1)
                iso_edge_attr = torch.tensor(iso_edge_attr, dtype=torch.long)

                # Sort indices.
                if iso_edge_index.numel() > 0:
                    perm = (iso_edge_index[0] * edge_attr.size(0) // 2 + iso_edge_index[1]).argsort()
                    iso_edge_index, iso_edge_attr = iso_edge_index[:, perm], iso_edge_attr[perm]

                pos = pos[:mol.GetNumAtoms()]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                            iso_edge_index=iso_edge_index, iso_edge_attr=iso_edge_attr,
                            pos=pos, y=y, smiles=smiles)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name][0]}({len(self)})'
