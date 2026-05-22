# Graph transformers — a brief primer

Background concepts for the rest of these docs. For the specific architecture used here, see [overview.md](overview.md#introduction-of-dgt) (DGT) and [tech.md](tech.md#model-architecture) (implementation).

## From MPNNs to graph transformers

The default workhorse for graph learning has been the **message-passing neural network (MPNN)**: at each layer, every node aggregates a function of its neighbours' features and updates itself. Stacking $L$ such layers lets information travel at most $L$ hops, and stacking deeper hits two well-known walls:

- **Over-smoothing** — node representations become indistinguishable after many message-passing rounds.
- **Over-squashing** — long-range dependencies are forced through narrow bottlenecks in the graph, losing signal.

**Graph transformers** were introduced to address both: replace (or supplement) local message passing with **self-attention over all node pairs**, so every node can in principle attend to every other node in one layer.

## What changes vs. a standard Transformer

A vanilla Transformer takes a *sequence* of tokens and relies on **positional encodings** (sinusoidal or learned) to inject order. Graphs have no canonical node order, so naïvely running self-attention on graph nodes throws away topology. Graph transformers fix this in two complementary ways:

1. **Positional / structural encodings (PE / SE).** Each node gets a feature vector summarising where it sits in the graph: Laplacian eigenvectors (LapPE), random-walk return probabilities (RWSE / RWPE), SignNet, etc. Pairwise encodings — shortest-path distance (SPDE), ring membership (RSE), or full distance matrices — are added as **attention biases** so the score between two nodes depends on how they are connected, not just on their features.
2. **Edge features as attention biases / value modulators.** Bond type, bond length, or any pairwise feature can be injected into the attention computation as an additive bias on the score and / or on the value. This is the mechanism DGT relies on (see the $E^{\text{att}}$ / $E^{\text{val}}$ terms in [tech.md](tech.md#model-architecture)).

Graph transformers preserve the transformer attention framework, but replace sequence-aware assumptions with graph-aware structural encodings. 
Graph-aware attention is commonly achieved by injecting structural information into attention through masks (e.g., adjacency matrix, 1 or 0), attention biases, edge embeddings/features, or structural encodings.

The core hidden-state shape is similar: [batch, items, hidden_dim].
Sequence transformer:
X: [batch_size, sequence_length, hidden_dim]
Graph transformer:
X: [batch_size, num_nodes, hidden_dim]
The difference is the extra structural information. For a sequence transformer, structure is usually simple:
position_ids: [B, N]
position_embeddings: [B, N, d]
causal/padding mask: [B, N] or [B, N, N]
For a graph transformer, structure is usually richer and pairwise:
adjacency: [B, N, N]
edge features: [B, N, N, d_e]
distance matrix: [B, N, N]
attention bias: [B, num_heads, N, N]

| Aspect | Conventional Transformer | Graph Transformer |
|---|---|---|
| Main data type | Sequence data, such as text tokens | Graph data, such as nodes and edges |
| Basic input unit | Token (with order as provision) | Node (no natural order)|
| Main input shape | `[B, L, d]` (B: batch size)| `[B, N, d]` |
| Meaning of length dimension | `L` = sequence length / number of tokens | `N` = number of nodes |
| Core framework | Token embeddings → Q/K/V → attention → FFN | Node embeddings → Q/K/V → graph-aware attention → FFN |
| Q, K, V computation | Computed from token embeddings | Computed from node embeddings |
| Attention target | Tokens attend to other tokens | Nodes attend to other nodes |
| Basic attention score shape | `[B, H, L, L]` (H: hidden dimension)| `[B, H, N, N]` |
| Structural information | Mainly token order and position | Graph topology, adjacency, edge type, distance, connectivity |
| Extra structural input | Position IDs or positional embeddings, e.g. `[B, L]` or `[B, L, d]` | Adjacency, edge features, distance matrix, attention bias |
| Common extra input shapes | Position: `[B, L]`; mask: `[B, L]` or `[B, L, L]` | Adjacency: `[B, N, N]`; edge features: `[B, N, N, d_e]`; bias: `[B, H, N, N]` |
| How structure is injected | Added to token embeddings or used as attention mask | Added as node/edge features, positional encodings, masks, or attention bias |
| Typical attention formula | `softmax(QKᵀ / sqrt(d_k))V` | `softmax(QKᵀ / sqrt(d_k) + B_graph)V` |
| Natural ordering | Yes, tokens have sequence order | No, graph nodes usually have no natural order |
| Masking | Often causal mask or padding mask | May use adjacency mask, padding mask, or full global attention |
| Connectivity assumption | Usually every token can attend to every previous/all token depending on task | Nodes may attend globally or only to graph neighbors |
| Output representation | Token-level representations, often pooled or decoded | Node-level representations, often pooled for graph-level tasks |
| Main difference | Learns relationships in ordered sequences | Learns relationships using graph-aware structural information |

## Local vs. global, and the hybrid recipe

Pure global attention loses the strong locality prior that makes MPNNs sample-efficient on chemistry-like graphs. The current dominant recipe — popularised by **GraphGPS** ([overview.md](overview.md#underlying-framework-graphgps)) — runs a local MPNN module and a global attention module **in parallel** at every layer and sums their outputs. The MPNN side enforces locality, the attention side mixes long-range information, and PE/SE encodings let both modules reason about graph structure.

## Why graph transformers matter for molecules

Molecules are small graphs (often <50 heavy atoms), so $O(N^2)$ attention is affordable, and they have long-range physicochemical interactions (conjugation, intramolecular H-bonding, ring effects) that pure short-stack MPNNs struggle to capture. Several recent state-of-the-art molecular property predictors — Graphormer, SAN, GraphGPS, GROVER, UniMol, and **DGT** in this repo — are graph transformers with progressively richer encodings.

## Key models in one line each

- **Graphormer** (Ying et al., NeurIPS 2021) — popularised SPDE as an attention bias on top of vanilla Transformer.
- **SAN** (Kreuzer et al., NeurIPS 2021) — learnable Laplacian PE with a Transformer encoder.
- **GraphGPS** (Rampášek et al., NeurIPS 2022) — modular MPNN + global-attention hybrid; the framework this repo builds on.
- **DGT** (this repo) — graph transformer run on a *dual* atom-graph / bond-graph view, with SPDE + RWPE + RSE pairwise encodings and optional Bessel / spherical-harmonic 3D terms.
