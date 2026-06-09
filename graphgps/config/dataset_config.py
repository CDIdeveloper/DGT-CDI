from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # Share edge attention biases and values between layers.
    cfg.dataset.edge_encoder_shared = False

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # Whether to add ring information per graph
    cfg.dataset.rings = False
    cfg.dataset.rings_max_length = 6
    cfg.dataset.rings_coalesce_edges = False

    # Whether to add shortest path information per graph
    cfg.dataset.spd = False
    cfg.dataset.spd_max_length = 6

    # Whether to complete dense edge features
    cfg.dataset.edge_encoder_dense = True

    # --- Molecular descriptors (Phase 2, late fusion at the head) ---
    # Dimensionality of the per-molecule descriptor vector (Data.desc).
    # 0 = no descriptors (baseline). Set to the dataset's desc_dim
    # (biodeg=216, biodeg_gwu=247) for the line_graph_with_desc head.
    cfg.dataset.desc_dim = 0

    # When True, the loader z-scores descriptors using TRAIN-split mean/std
    # and writes a SEPARATE processed cache (the baseline raw-desc cache is
    # preserved). Stats + column names are persisted to desc_stats.json.
    cfg.dataset.standardize_desc = False

    # Descriptor-column selection (Phase 2 descriptor-type study). Precedence:
    # desc_columns (explicit) > desc_include/desc_exclude (substrings) > all.
    # Any non-empty selection -> its own processed cache (keyed by a hash of the
    # resolved columns). Set dataset.desc_dim to the SELECTED count.
    cfg.dataset.desc_include = []   # keep columns containing ANY substring (e.g. ['_gwu'])
    cfg.dataset.desc_exclude = []   # drop columns containing ANY substring
    cfg.dataset.desc_columns = []   # explicit exact column names (subset)
