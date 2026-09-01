import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.dataset.name in ['MNIST', 'CIFAR10']:
            self.in_dim = 1
        elif cfg.dataset.format == 'PyG-MoleculeNet':
            self.in_dim = cfg.dataset.edge_encoder_num_types
        elif cfg.dataset.format == 'PyG-QM9':
            self.in_dim = cfg.dataset.edge_encoder_num_types
        elif cfg.dataset.format == 'PyG-Chiral3DMoleculeNet':
            self.in_dim = cfg.dataset.edge_encoder_num_types
        elif cfg.dataset.format == 'PyG-biodeg_gwu':
            # biodeg_gwu uses MoleculeNet-identical edge featurisation
            # (3 categorical bond features: bond_type, stereo, is_conjugated).
            self.in_dim = cfg.dataset.edge_encoder_num_types
        elif cfg.dataset.format == 'PyG-biodeg_gwu_no_ind':
            # biodeg_gwu_no_ind (InD rows removed) — same featurisation.
            self.in_dim = cfg.dataset.edge_encoder_num_types
        elif cfg.dataset.format == 'PyG-biodeg':
            # biodeg (no-Reaxys) — same MoleculeNet-identical edge featurisation.
            self.in_dim = cfg.dataset.edge_encoder_num_types
        elif cfg.dataset.format == 'OGB':
            self.in_dim = cfg.dataset.edge_encoder_num_types
        else:
            raise ValueError("Input edge feature dim is required to be hardset "
                             "or refactored to use a cfg option.")
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch
