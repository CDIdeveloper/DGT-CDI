# Dual Graph Transformer for Molecular Property Prediction

This repo is the implementation of the dual graph transformer for molecular property prediction.
This model integrates atom and bond graphs to encode the comprehensive molecular information, including atom and bond features, graph topology and structure, and 3D spatial information if available for enhanced molecular property prediction performance.

![dgt](./imgs/dgt.png)

To install, run the following commands in sequence
```
conda create -n dgt python=3.10.16
conda activate dgt
conda install mamba
mamba install graph-tool==2.45
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric==2.0.4
pip install torchmetrics==1.2.0
pip install ogb==1.3.6
pip install tensorboardX==2.6.2.2
pip install wandb==0.18.7
pip install rdkit==2025.9.1
pip install libauc==1.1.0
```

Since molecular SMILES is not provided along with the download link from the `torch_geometric` package for the QM9 dataset. We provide that in the `datasets/QM9_split` folder, together with the scaffold splitting results. After downloading the QM dataset, please copy files under `datasets/QM9_split` to `datasets/QM9/raw`.  

Config files for reproducing our results are provided in the `configs` folder.  
To train from scratch, run `python main.py --cfg config_file_path --repeat 1 seed 0 wandb.use False`.  

This implementation is developed from graphgps. For more information, please check [https://github.com/rampasek/GraphGPS.git](https://github.com/rampasek/GraphGPS.git)

---

## DGT-CDI fork — CDI datasets and workflow

The sections above cover the public benchmark datasets (QM9, BBBP, ...). This fork adds the CDI biodegradability pipeline: S3-hosted datasets, molecular-descriptor late fusion, and a train/val/test protocol that holds the test set out until the very end. Those datasets are **not** downloaded by `torch_geometric` — they are fetched once from S3 by a prepare script.

| Task | Start here |
|---|---|
| **Onboard new data from S3** (new drop, or existing data under a new name) | [documents/modeling_routine.md → Step 0](documents/modeling_routine.md#step-0--onboard-a-new-dataset-new-data-on-s3) |
| Train → deployable model → predict, end to end | [documents/modeling_routine.md → Quickstart](documents/modeling_routine.md#quickstart--end-to-end-training-routine) |
| Which YAML key does what | [documents/config_reference.md](documents/config_reference.md) |
| Leak-free selection protocol + leakage audit | [documents/dgt_porting_guide.md](documents/dgt_porting_guide.md) §2, §7 |
| What this fork changed vs. the original DGT | [documents/upstream_sync.md](documents/upstream_sync.md) |

Two things that bite newcomers:

- **`scripts/prepare_data.py` needs a different conda env** than training (`boto3`, `python-dotenv`, `pyarrow` — not in `dgt`). Run it from trans_learn's env, then switch back. Everything after it is offline.
- **Re-running `prepare_data.py` under an existing dataset name does not refresh the cache.** PyG re-processes only when the processed file is missing, so you must `rm -rf datasets/<name>/processed/` or you will silently train on the previous data.

For the training pipeline itself, `python main.py --cfg <config> --repeat 4 seed 0 wandb.use False` works the same as above; CDI configs live in [configs/biodegradability/](configs/biodegradability/) and use `train.mode: dgt` (test scored once, on the best-val checkpoint).