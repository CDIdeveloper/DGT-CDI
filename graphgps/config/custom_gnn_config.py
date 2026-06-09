from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False

    # Output dim of the descriptor projection MLP f(desc) in the
    # `line_graph_with_desc` head (Phase 2). Tunable knob to modulate the
    # molecular-descriptor channel's width/influence; model-only (no cache
    # invalidation when swept). Only used when gnn.head = line_graph_with_desc.
    cfg.gnn.desc_proj_dim = 128
