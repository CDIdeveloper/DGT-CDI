"""Prepare a trans_learn dataset for DGT training (offline, one-shot).

Fetches one dataset from trans_learn's ``DATASET_REGISTRY`` (via S3), runs
trans_learn's preprocessing (NaN-fill, drop-columns, target-NaN filter; no
normalisation — descriptor normalisation belongs to Phase 2), and writes a
local snapshot for the DGT PyG loader.

The DGT training loop reads ONLY the local files written here, so no AWS
credentials are needed at training time. Re-run this script only when you
bump dataset versions on S3.

CLI
---
    python scripts/prepare_data.py \\
        --dataset            <name>          (required; key in DATASET_REGISTRY)
        --trans-learn-path   <path>          (required; the trans_learn repo root)
        [--output-dir        <path>]         (default: datasets/<dataset>/raw/)
        [--smiles-col        <name>]         (default: auto-detect 'smiles')

Example
-------
On the lab box where trans_learn lives at ``/home/jovyan/tools/trans_learn``::

    python scripts/prepare_data.py \\
        --dataset biodeg_gwu \\
        --trans-learn-path /home/jovyan/tools/trans_learn

This produces::

    datasets/biodeg_gwu/raw/train.parquet
    datasets/biodeg_gwu/raw/test.parquet
    datasets/biodeg_gwu/raw/manifest.json

The PyG loader (Phase 1) reads these three files; ``manifest.json`` carries
``id_column_count`` / ``target_column`` / ``smiles_column`` /
``descriptor_columns`` / ``desc_dim`` / ``task_type_hint`` so the loader
does not need to import trans_learn at training time.

Output layout
-------------
- ``train.parquet`` / ``test.parquet`` keep the trans_learn column layout:
  the first ``id_column_count`` columns hold SMILES + identifiers + the
  target ``y``; remaining columns are descriptors.
- ``manifest.json`` is the contract between this script and the PyG loader.

Environment requirements
------------------------
- **Python env**: needs ``python-dotenv``, ``boto3``, ``pandas``, ``pyarrow``.
  The ``dgt`` conda env does NOT have these — activate trans_learn's env
  (or any env with those packages) before running this script, then switch
  back to ``dgt`` for training.

- **AWS credentials**: ``boto3`` reads ``~/.aws/credentials`` or ``AWS_*``
  env vars in the standard way.

- **trans_learn ``.env``**: ``trans_learn.settings`` calls ``find_dotenv()``,
  which searches upward from CWD. Easiest setup: run this script from
  inside the trans_learn checkout, OR copy trans_learn's ``.env`` into the
  DGT-CDI repo root, OR set ``ENV_DEVELOPMENT`` in your shell before
  running. Without it, the import will fail with a clear ``FileNotFoundError``.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_smiles_column(id_columns):
    """Auto-detect the SMILES column (case-insensitive substring match)."""
    candidates = [c for c in id_columns if 'smiles' in c.lower()]
    if not candidates:
        raise ValueError(
            f"Could not auto-detect SMILES column in {list(id_columns)}. "
            f"Pass --smiles-col <name> explicitly."
        )
    if len(candidates) > 1:
        exact = [c for c in candidates if c.lower() == 'smiles']
        if len(exact) == 1:
            return exact[0]
        raise ValueError(
            f"Multiple SMILES-like columns matched: {candidates}. "
            f"Pass --smiles-col <name> explicitly."
        )
    return candidates[0]


def _summarize_target(series, target_column):
    """Print a short summary of the target series; return a task_type hint."""
    n_unique = series.nunique()
    if n_unique <= 5 and pd.api.types.is_numeric_dtype(series):
        counts = series.value_counts().sort_index()
        print(f"  target '{target_column}': {dict(counts)}")
        return ('classification_binary' if n_unique == 2
                else 'classification_multiclass')
    if pd.api.types.is_numeric_dtype(series):
        print(f"  target '{target_column}': "
              f"mean={series.mean():.4f}, std={series.std():.4f}, "
              f"min={series.min():.4f}, max={series.max():.4f}")
        return 'regression'
    print(f"  target '{target_column}': {n_unique} unique values; "
          f"sample: {series.head().tolist()}")
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(
        description="Fetch & preprocess a trans_learn dataset for DGT training.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name registered in trans_learn's DATASET_REGISTRY "
             "(e.g. biodeg_gwu, biodeg, abs, emi, qy, extin).",
    )
    parser.add_argument(
        "--trans-learn-path", type=Path, required=True,
        help="Filesystem path to the trans_learn repo root (the parent of "
             "the trans_learn/ package directory; e.g. "
             "/home/jovyan/tools/trans_learn).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write the local snapshot. "
             "Default: datasets/<dataset>/raw/",
    )
    parser.add_argument(
        "--smiles-col", type=str, default=None,
        help="SMILES column name. Default: auto-detect by case-insensitive "
             "'smiles' substring match within the id columns.",
    )
    args = parser.parse_args()

    # 1. Wire up sys.path so trans_learn (provides DATASET_REGISTRY, S3Handler)
    # and the local tests/data_loading/load_data.py (DatasetLoader) are both
    # importable. Deferred import is needed here because the trans_learn path
    # comes from CLI input — module-level imports would fail before resolution.
    tl_path = args.trans_learn_path.resolve()
    if not tl_path.is_dir():
        raise FileNotFoundError(f"--trans-learn-path not found: {tl_path}")
    sys.path.insert(0, str(tl_path))
    sys.path.insert(0, str(REPO_ROOT / 'tests' / 'data_loading'))
    try:
        from load_data import DatasetLoader  # noqa: E402
    except Exception as e:
        raise RuntimeError(
            "Failed to import DatasetLoader (tests/data_loading/load_data.py)."
            "\nLikely causes:"
            "\n  (1) trans_learn deps missing in this Python env"
            " (python-dotenv, boto3, pandas, pyarrow)."
            "\n  (2) trans_learn's .env not discoverable from CWD"
            " (find_dotenv() looks upward)."
            f"\n  (3) --trans-learn-path is wrong: {tl_path}"
            f"\nUnderlying error: {e}"
        ) from e

    # 2. Fetch + preprocess via trans_learn's DatasetLoader.
    # normlize=False on purpose: descriptor standardisation is Phase 2's job
    # (z-score using train-set mean/std, persisted for test/val to reuse).
    print(f"Loading dataset '{args.dataset}' from S3 via trans_learn ...")
    loader = DatasetLoader(args.dataset)
    train_data = loader.load_split('train', normlize=False, load_fps=False)
    test_data = loader.load_split('test', normlize=False, load_fps=False)

    # 3. Reconstruct full DataFrames (concat ids_ys + descs back into one).
    df_train = pd.concat([train_data['ids_ys'], train_data['descs']], axis=1)
    df_test = pd.concat([test_data['ids_ys'], test_data['descs']], axis=1)

    id_column_count = loader.config.id_column_count
    target_column = loader.config.target_column
    id_columns = list(df_train.columns[:id_column_count])
    descriptor_columns = list(df_train.columns[id_column_count:])

    if target_column not in id_columns:
        raise KeyError(
            f"target_column '{target_column}' not found in the first "
            f"id_column_count={id_column_count} columns "
            f"({id_columns}). Check the DATASET_REGISTRY entry "
            f"for '{args.dataset}'."
        )

    smiles_col = args.smiles_col or _find_smiles_column(id_columns)
    if smiles_col not in id_columns:
        raise KeyError(
            f"--smiles-col '{smiles_col}' not in id columns {id_columns}."
        )

    # 4. Write parquets + manifest.
    out_dir = (args.output_dir or
               (REPO_ROOT / 'datasets' / args.dataset / 'raw')).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / 'train.parquet'
    test_path = out_dir / 'test.parquet'
    manifest_path = out_dir / 'manifest.json'

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    print(f"\nTrain ({len(df_train)} rows):")
    task_type_hint = _summarize_target(df_train[target_column], target_column)
    print(f"\nTest ({len(df_test)} rows):")
    _summarize_target(df_test[target_column], target_column)

    manifest = {
        'dataset_name': args.dataset,
        'id_column_count': id_column_count,
        'id_columns': id_columns,
        'target_column': target_column,
        'smiles_column': smiles_col,
        'descriptor_columns': descriptor_columns,
        'desc_dim': len(descriptor_columns),
        'n_train': int(len(df_train)),
        'n_test': int(len(df_test)),
        'task_type_hint': task_type_hint,
    }
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nWrote:\n  {train_path}\n  {test_path}\n  {manifest_path}")
    print(f"\ndesc_dim       = {manifest['desc_dim']}  "
          f"(→ dataset.desc_dim in YAML)")
    print(f"task_type_hint = {manifest['task_type_hint']}  "
          f"(→ dataset.task_type in YAML)")


if __name__ == '__main__':
    main()
