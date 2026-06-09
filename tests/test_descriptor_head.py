"""Minimal unit test for the descriptor-fusion head (`line_graph_with_desc`).

Runs on CPU in seconds (no dataset, no GPU). Verifies:
  - both `line_graph` and `line_graph_with_desc` heads are registered;
  - `line_graph_with_desc` does a forward pass on a toy batch and returns
    logits of shape `[B, dim_out]`, fusing the molecular-descriptor channel
    (`batch.desc`) after the readout.

This is the single unit test for Phase 2 (per the pr-1 log testing decision);
broader checking is left to the end-to-end 3-epoch dry-run + full run.

Run: pytest -q tests/test_descriptor_head.py
"""
from types import SimpleNamespace

import torch

import graphgps  # noqa: F401 — registers heads + config in the GraphGym registry
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.register import head_dict


def _toy_batch(n_atoms_per_graph, n_bonds_per_graph, dim_in, desc_dim):
    """Build a minimal stand-in for a collated DGT batch."""
    batch_idx, e_batch = [], []
    for g, n in enumerate(n_atoms_per_graph):
        batch_idx += [g] * n
    for g, m in enumerate(n_bonds_per_graph):
        e_batch += [g] * m
    B = len(n_atoms_per_graph)
    return SimpleNamespace(
        x=torch.randn(sum(n_atoms_per_graph), dim_in),
        e=torch.randn(sum(n_bonds_per_graph), dim_in),
        batch=torch.tensor(batch_idx, dtype=torch.long),
        e_batch=torch.tensor(e_batch, dtype=torch.long),
        desc=torch.randn(B, desc_dim),
        y=torch.randint(0, 2, (B, 1)).float(),
    )


def test_heads_registered():
    assert 'line_graph' in head_dict, "baseline line_graph head missing"
    assert 'line_graph_with_desc' in head_dict, \
        "descriptor-fusion head not registered"


def test_descriptor_head_forward():
    dim_in, dim_out, desc_dim, desc_proj_dim = 8, 1, 5, 4
    set_cfg(cfg)
    cfg.dataset.desc_dim = desc_dim
    cfg.gnn.desc_proj_dim = desc_proj_dim
    cfg.gnn.act = 'gelu'
    cfg.model.graph_pooling = 'add'

    head = head_dict['line_graph_with_desc'](dim_in=dim_in, dim_out=dim_out)
    batch = _toy_batch([3, 2], [4, 2], dim_in, desc_dim)  # B=2 graphs
    pred, label = head(batch)

    assert pred.shape == (2, dim_out), f"unexpected logits shape {pred.shape}"
    assert label.shape == (2, 1)
    assert torch.isfinite(pred).all(), "non-finite logits"
