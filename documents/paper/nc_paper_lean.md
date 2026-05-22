# Introduction of DGT

## Background

Employing the self-attention mechanism on molecules requires the comprehensive encoding of molecular information, including atom and bond features, graph topology and structure, and three-dimensional (3D) information if available, in the query-key-value operation [17, 18]. Graph-specific encodings — including relative positional encoding (RPE) [19] and ring structural encoding (RSE) [20] — can be incorporated into the graph transformer layer, along with edge features, to capture complex dependencies within molecules [21]. In some graph transformer implementations, edge features are also aggregated as characterisation of local neighbourhoods for updating node features before the attention calculation [22]. Despite that molecules are often intuitively depicted as topological graphs with atoms as nodes, reinterpreting bonds as nodes in molecular graphs can provide an alternative perspective to enrich molecular representation in the attention mechanism and thereby promote property prediction [23].

To this end, we propose the **dual graph transformer (DGT)**, a novel deep learning architecture that leverages the self-attention mechanism for enhancing molecular property prediction by integrating the atom graph with its bond counterpart. Atom and bond features are mutually fused into the attention score matrices of their counterparts. RPE and RSE are incorporated into the attention matrices of both atom and bond graphs. DGT also enables encoding bond lengths, atom–atom distances, and bond–bond angles into the self-attention mechanism for the optional adoption of 3D information of different fidelities.

## Overview of DGT

The proposed DGT is illustrated in Fig. 1, encompassing stages of graph representation, molecule encoding, stacked graph transformer layers, and a final readout module for molecular property prediction. The bond graph representation is derived by reinterpreting the intuitive atom graph with the line graph transform, which exchanges roles of atoms and bonds in the topological structure. This provides an alternative perspective to enrich molecular features, thereby enhancing the property prediction performance of deep learning approaches. As the exemplified phenyl formate in Fig. 1a, carbon and oxygen atoms are regarded as edges in the bond graph representation, depicting connections of single, double, and aromatic bonds. The highlighted ipso carbon in the atom graph corresponds to the three highlighted edges, sharing the same feature vector, in the bond graph.

Such a dual graph representation realises the encoding of comprehensive molecular information — including atom and bond features, graph topology, specific structures, and 3D information if available — into deep learning architectures for molecular learning tasks. In detail, atom and bond features are represented as node vectors in their respective graphs and as edge vectors in their mutual graphs. Shortest path distance encoding (SPDE) and random walk positional encoding (RWPE) are employed as the RPE of nodes; RSE is added to deliver ring information, given its importance for many molecular properties. Together, these graph-specific encodings are combined with edge features to form node–node matrices as pairwise features for both atom and bond graphs. Additionally, this dual graph representation accommodates diverse 3D information: bond lengths can be embedded within bond features; atom–atom distances and bond–bond angles can be fit into the pairwise feature matrices of atom and bond graphs, respectively.

The graph transformer layer extends the conventional query-key-value operation by merging the pairwise feature matrices from atom and bond graphs into their respective attention score matrices and passing messages, allowing DGT to learn both local and global dependencies for atoms and bonds (Fig. 1b). In addition, the pairwise feature matrices are also utilised to update node vectors within the self-attention calculation, providing an effective means to enhance node messages, as demonstrated in recent studies [24, 25]. Batch normalisation is applied in each graph transformer layer to stabilise training. Details of the graph transformer layer are provided in the Methods section. After stacked graph transformer layers, the global average pooling is employed as a readout function, followed by the concatenation operation of outputs of both graph representations and fully connected layers for molecular property prediction.

**Fig. 1 — An illustration of DGT.**
**a**, Atom and bond graph representations of molecules. Positions of atoms and bonds are exchanged in the dual graph representation, accommodating atom and bond features, relative position, and structural information. Bond length, atom–atom distance, and bond–bond angle can also be integrated into DGT when available.
**b**, Molecule encodings and graph transformer structure. DGT encodes molecules into atom and bond vectors, atom–atom matrices, and bond–bond matrices. Graph transformer layers are stacked to conduct the query-key-value operation, followed by the pooling layer for predicting molecular properties.

# Methods

## Datasets and splitting strategy

We thoroughly evaluated DGT across four domains for molecular property prediction, encompassing ten datasets and a total of 58 subtasks from the MoleculeNet benchmark [26]. Datasets included in the physiology were BBBP, ClinTox, and Tox21, which comprises 12 classification tasks ranging from NR-AR to SR-p53, as well as SIDER, which covers 27 tasks related to adverse drug reactions. BACE and HIV were selected from the biophysics domain to assess classification performance on drug activity and antiviral screening, respectively. For physical chemistry, three regression tasks were employed: ESOL, FreeSolv, and Lipophilicity. Finally, the quantum chemistry domain was represented by the QM9 dataset, comprising 12 regression tasks targeting molecular quantum properties, ranging from basic properties such as dipole moments to electronic properties including frontier orbital energies. Randomly sampled subsets of the PCQM4Mv2 dataset were utilised as the training set for investigating the pretraining of DGT(3D).

We employed the scaffold split [50] method to partition datasets from the MoleculeNet benchmark into training, validation, and test sets with an 80:10:10 ratio. In this approach, splitting is based on the molecular scaffold — essentially the core structure of a molecule, essentially the fundamental ring system or backbone without any substituent groups. Compared to the random split, the scaffold split is more challenging by ensuring that the test set contains molecules with scaffolds that were not encountered during training, thereby requiring the model to generalise to new chemical structures. To ensure a fair comparison between DGT and baseline models, the model training in this work was repeated four times with the same data split to get the average result. The scaffold split method was implemented following the publicly available codes from the chemprop GitHub repository (https://github.com/aamini/chemprop.git).

### Physiology

- **BBBP** — 2,039 molecules annotated with binary labels indicating whether each compound can penetrate the blood–brain barrier, a key consideration in central nervous system drug development.
- **ClinTox** — 1,478 compounds categorised according to whether they have been approved or withdrawn from clinical trials due to severe toxicity.
- **Tox21** — a public database consisting of 7,831 instances qualitatively evaluated across 12 biological targets, including nuclear receptors and stress response pathways, providing a diverse set of multi-label toxicity prediction tasks.
- **SIDER** — 1,427 marketed drugs mapped to 27 types of clinically reported adverse drug reactions, supporting the association between molecular structures and side-effect profiles for developing safer pharmaceuticals.

### Biophysics

- **BACE** — 1,513 small-molecule inhibitors annotated for their binding activity toward the human β-secretase 1 enzyme, a critical target in Alzheimer's disease research.
- **HIV** — 41,127 compounds providing profound data on molecules' ability to inhibit HIV replication, offering a large-scale benchmark for antiviral activity classification.

### Physical chemistry

- **ESOL** — 1,128 molecules with experimentally measured aqueous solubility values, supporting the investigation of how molecular features influence solubility.
- **FreeSolv** — a curated collection of 642 small organic compounds with hydration free energies, serving as a data source for studying drug solubility and stability.
- **Lipophilicity** — 4,200 compounds with experimental octanol–water partition coefficients, describing how well a substance interacts with non-polar environments and offering essential data for analysing pharmacokinetic behaviour.

### Quantum mechanics

- **QM9** — 133,885 small molecules composed of up to nine heavy atoms (C, O, N, and F), each annotated with quantum mechanical properties computed at the B3LYP/6-31G(2df,p) level of theory. The dataset includes 12 regression targets capturing fundamental quantum molecular properties crucial for versatile domains: dipole moment, polarisability, squared radius, zero-point vibrational energy, heat capacity at constant volume, highest occupied molecular orbital (HOMO) energy, lowest unoccupied molecular orbital (LUMO) energy, HOMO–LUMO energy gap, internal energy at 0 K, internal energy at standard state, enthalpy, and Gibbs free energy.

## Experimental settings

The experimental setting section consists of evaluation metrics and pretraining setup.

### Evaluation metrics

We benchmarked DGT across 58 molecular property prediction tasks spanning four distinct domains, each evaluated using domain-appropriate metrics.

- **AUC-ROC** — used for classification tasks within the physiology and biophysics domains, as it quantifies a model's ability to distinguish between classes across various threshold settings; a higher AUC-ROC value indicates better classification performance.
- **RMSE** — adopted for physical chemistry to assess the difference between predicted and true values.
- **MAE** — employed for the quantum mechanics domain as a routine metric to measure the average magnitude of prediction errors.

### Pretraining setup

We investigated the applicability of the pretraining strategy on DGT(3D) using the large-scale PCQM4Mv2 dataset [36]. In this setup, DGT(3D) was first pretrained on the PCQM4Mv2 dataset to predict the HOMO–LUMO energy gap, a relevant task intended to enhance the downstream prediction of HOMO and LUMO energies within the QM9 dataset. To assess the impact of the pretraining dataset size, we carried out a series of pretraining runs using subsets with 10K, 100K, and 1M molecules randomly sampled from PCQM4Mv2. DGT(3D) was pretrained on these sampled molecules for 100 epochs incorporating a linear warmup strategy over the first 5 epochs, using the AdamW optimiser with a batch size of 128 and a small learning rate of 0.0002, scheduled via cosine annealing. The resulting pretrained weights were then utilised to initialise DGT(3D) for training on the QM9 dataset. Specifically, PCQM4Mv2 involves a total of 22 elements, covering those present in the QM9 dataset (H, C, O, N, and F); relevant atomic embeddings were accordingly extracted and concatenated to initialise the atom embedding matrix for QM9.

## Self-attention modules

For molecular property prediction, DGT encodes the tuples $G^a = (N^a, E^a)$ and $G^b = (N^b, E^b)$ as the atom and bond graphs, respectively, where:

- $N^a$ is the set of atoms,
- $E^a \subseteq N^a \times N^a$ is the set of directed bonds,
- $N^b$ is the set of undirected bonds,
- $E^b \subseteq N^b \times N^b$ is the set of neighbour pairs of undirected bonds.

The numbers of atoms and bonds are denoted $|N^a|$ and $|N^b|$, respectively. Given the atom feature dimension size $d_a$, atom features are encoded as rows in a matrix $N^a \in \mathbb{R}^{|N^a| \times d_a}$; likewise, bond features are encoded as $N^b \in \mathbb{R}^{|N^b| \times d_b}$, where $d_b$ is the bond feature dimension size.

The biased multi-head attention mechanism is formulated as:

$$
s_{i,j} \;\propto\; \exp\!\left( \frac{Q_i \, K_j^{\top}}{\sqrt{d_k}} + E^{\text{att}}_{i,j} \right)
$$

$$
h_i \;=\; \sum_{j} s_{i,j} \left( V_j + E^{\text{val}}_{i,j} \right)
$$

where $Q$, $K$, and $V$ are respective projections of $X$ (either $N^a$ or $N^b$) for query, key, and value; $E^{\text{att}}$ and $E^{\text{val}}$ are respective projections of the node–node encoding $E$ (either $E^a$ or $E^b$) to attention bias and edge feature injection. $d_k$ denotes the dimensionality of the key vectors and is involved for stabilising the training process. $h_i$ is the hidden vector for node $i$.

The node–node encoding $E$ is the sum of encodings of relative position and ring structure. Notably, DGT(3D) enables incorporating bond lengths into the node feature matrix $N^b$, atom–atom distances into $E^a$, and bond–bond angles into $E^b$. To inform the model with enriched local geometric representation, interatomic distances are expanded using the physics-inspired orthogonal Bessel basis functions to improve expressivity and stability; bond angular information is encoded with spherical harmonics functions defined on the unit sphere to ensure rotational consistency [51]. Such geometry-aware representations are crucial for accurate property prediction.

With this attention calculation hereafter referred to as $\text{MHA}(X, E^{\text{att}}, E^{\text{val}})$, the DGT layer can be given as:

$$
H \;=\; X + \text{MLP}\!\left(\text{MHA}(X, E^{\text{att}}, E^{\text{val}})\right)
$$

$$
X \;=\; \text{BatchNorm}\!\left( \text{FFN}(H) \right)
$$

Here, the input hidden node feature matrix $X$ is added to $\text{MHA}(X, E^{\text{att}}, E^{\text{val}})$ transformed by the multilayer perceptron (MLP). A feed-forward network (FFN) follows to implement nonlinear transformations to each token embedding to obtain the output node feature matrix $X$. Batch normalisation (BatchNorm) is applied to avoid overfitting.

For the readout part, atom and bond features are globally averaged and concatenated for predicting the molecular property as:

$$
\text{MLP}\!\left( \text{GAP}(X^a) \,\Vert\, \text{GAP}(X^b) \right)
$$

where GAP is the global average pooling operation and the Rectified Linear Unit (ReLU) activation function is attached to linear layers of MLP.

## Baselines

DGT was comprehensively compared with state-of-the-art machine learning methods on datasets from the MoleculeNet benchmark.

- **Random forest (RF)** — adopted as a baseline machine learning model [26]. RF employs **ECFP4** as input features, a widely adopted circular fingerprint in cheminformatics and molecular machine learning, which encodes molecules by capturing local atom-centred substructures.
- **Attentive FP** [27] — extends conventional GNNs by incorporating the attention mechanism that dynamically adjusts the importance weights of neighbouring atoms and bonds during message passing.
- **D-MPNN** [28] — refines standard message passing neural networks by directing messages over bonds instead of atoms, improving the encoding of bond-level interactions critical for accurate molecular property prediction.
- **Pretrained GNN** [29] — demonstrates the performance gains of a GNN architecture by informing it with large-scale molecular datasets before fine-tuning on downstream tasks.
- **GraphMVP** [30] — leverages a multi-view contrastive learning strategy, aligning information from the 2D molecular graph and its corresponding 3D conformer to encourage representations consistent across different structural modalities.
- **MolCLR** [31] — advances contrastive learning by generating diverse graph augmentations to improve the robustness of learnt features against input perturbations.
- **MoleBERT** [32] — adapts language model pretraining principles to molecular graphs by introducing masked atom and bond prediction tasks, facilitating versatile graph-level encoders that generalise across molecular prediction tasks.
- **GROVER** [33] — combines self-supervised learning objectives with a graph transformer backbone, learning rich contextualised molecular representations from millions of unlabelled molecules; presented as a representation model for the graph transformer.
- **UniMol** [34] — included as a representation of transformer-based 3D geometric deep learning models for molecular property prediction.

Performances of GraphMVP, MolCLR, MoleBERT, GROVER, and UniMol are derived from ref. [52] with the same data splitting. Attentive FP and D-MPNN are implemented using the DeepChem package (https://github.com/deepchem/deepchem.git); random forest is based on scikit-learn (https://github.com/scikit-learn/scikit-learn.git); and pretrained GNN is reproduced according to https://github.com/snap-stanford/pretrain-gnns.git.

For 3D molecular learning, DGT(3D) was benchmarked against:

- **SchNet** [37] — one of the earliest deep learning architectures explicitly designed for molecules and materials, incorporating continuous-filter convolutional layers to directly model quantum interactions. By treating interatomic distances as continuous inputs, SchNet effectively captures long-range dependencies for predicting quantum mechanical properties.
- **DimeNet** [38] — introduces directional message passing by encoding angular information between bonded atoms, enabling the model to learn higher-order geometric interactions.
- **SphereNet** [39] — extends this paradigm by leveraging spherical message passing, integrating both angular and radial basis functions to achieve a more complete characterisation of local geometric environments.
- **SE(3) Transformer** [40] — generalises attention mechanisms to 3D Euclidean space by enforcing equivariance under SE(3) group transformations, allowing molecular representations invariant to rotations and translations while preserving geometric symmetries.
- **GraphMVP** [30] and **3DInfomax** [41] — leverage 3D information to enrich 2D molecular embedding.

Packages for implementing these compared models:

- SchNet and DimeNet — https://github.com/pyg-team/pytorch_geometric.git
- SphereNet — https://github.com/divelab/DIG.git
- SE(3) Transformer — https://github.com/chao1224/Geom3D.git

## Data availability

Datasets used for benchmarking the DGT model are available at https://moleculenet.org/datasets. The PCQM4Mv2 dataset, used in pretraining DGT(3D), is accessible via https://ogb.stanford.edu/docs/lsc/pcqm4mv2. Source data are provided with this paper.

# Nomenclature

- **RPE** — relative positional encoding
- **RSE** — ring structural encoding
