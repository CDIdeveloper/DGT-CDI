"""Dataset-level tests for biodegradability loaders (and any future
PyG ``InMemoryDataset`` written by this fork).

Currently contains the **smoke / summary test** (``test_dataset_loads``).
Add new tests in this file as more dataset-level checks become useful —
e.g. cross-loader byte-identical featurisation, descriptor-column ordering
invariants, etc.

For each registered dataset, ``test_dataset_loads``:

1. Imports the loader class and instantiates it at the given root path.
   (PyG ``InMemoryDataset`` triggers ``process()`` on first run; subsequent
   invocations load the cached ``processed/data.pt`` and are fast.)
2. Asserts a set of structural invariants (splits non-empty, MoleculeNet-style
   atom featurisation, descriptor shape ``[1, desc_dim]``, descriptors finite).
3. **Prints a human-readable summary** of total / per-split counts,
   positive-class fractions, ``desc_dim``, and first-molecule tensor shapes.

The summary is the main artifact — ``pytest -rP`` shows captured stdout for
passed tests; pipe to a report file to keep a record.

Usage
-----

Run on the remote where the ``dgt`` env + ``datasets/<name>/processed/``
exist. Typical invocations::

    # All registered datasets, summary saved to a report file:
    pytest -rP tests/test_dataset.py > tests/report_dataset.txt

    # Single dataset (parametrize id matches the entry's first element
    # in DATASETS below):
    pytest -rP "tests/test_dataset.py::test_dataset_loads[biodeg_gwu]" \
        > tests/report_biodeg_gwu.txt

    pytest -rP "tests/test_dataset.py::test_dataset_loads[biodeg]" \
        > tests/report_biodeg.txt

Notes:
    - ``-rP`` shows captured stdout for PASSED tests (so the printed summary
      appears in the report). Without ``-rP``, pytest swallows prints.
    - Skipped if ``datasets/<name>/raw/`` does not exist on this machine
      (i.e., the prepare script has not been run). Run the prepare script
      first: ``python scripts/prepare_data.py --dataset <name>
      --trans-learn-path <path>``.

Adding a new dataset
--------------------

Append one tuple to ``DATASETS``::

    ("<dataset_id>", "<loader_module_dotted_path>", "<class_name>",
     "<dataset_root_path>")

The new entry is picked up automatically — no other edits needed.
"""
import importlib
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# Each tuple: (dataset_id_used_as_pytest_parametrize_id,
#              loader module (dotted),
#              loader class name,
#              dataset root path containing raw/ + processed/)
DATASETS = [
    (
        "biodeg_gwu",
        "graphgps.loader.dataset.biodeg_gwu",
        "BiodegGwu",
        "datasets/biodeg_gwu",
    ),
    (
        "biodeg",
        "graphgps.loader.dataset.biodeg",
        "Biodeg",
        "datasets/biodeg",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def _register_graphgps():
    """Import the graphgps package once per module so the GraphGym registry
    is populated (decorator side effects) before any dataset class is built."""
    import graphgps  # noqa: F401


@pytest.mark.parametrize(
    "loader_module, class_name, root",
    [(m, c, r) for _, m, c, r in DATASETS],
    ids=[d for d, *_ in DATASETS],
)
def test_dataset_loads(loader_module, class_name, root):
    """Load the dataset, check structural invariants, print a summary.

    Invariants asserted:
        - dataset is non-empty
        - each of train/val/test splits has > 0 entries
        - positive-class fraction per split is finite and in [0, 1]
        - first Data object has descriptor of shape [1, desc_dim] with
          desc_dim > 0 and finite values
        - first Data object has atom features x of shape [N, 9]
          (MoleculeNet-style featurisation) and edge_index of shape [2, ...]
    """
    root_path = REPO_ROOT / root
    if not (root_path / "raw").is_dir():
        pytest.skip(
            f"{root}/raw/ not found at {root_path}; run the prepare script:\n"
            f"  python scripts/prepare_data.py --dataset {root_path.name} "
            f"--trans-learn-path <path>"
        )

    cls = getattr(importlib.import_module(loader_module), class_name)
    ds = cls(str(root_path))

    n_total = len(ds)
    assert n_total > 0, f"empty dataset at {root_path}"

    splits = Counter(d.split for d in ds)
    pos = Counter(d.split for d in ds if int(d.y.item()) == 1)

    print()
    print(f"=== {class_name}({root}) — smoke summary ===")
    print(f"Total molecules: {n_total}")
    for s in ("train", "val", "test"):
        n = splits[s]
        p = pos[s]
        frac = (p / n) if n else 0.0
        print(f"  {s}: n={n}, positives={p} ({frac:.1%}), negatives={n - p}")
        assert n > 0, f"empty {s} split"
        assert 0.0 <= frac <= 1.0

    first = ds[0]
    desc_dim = first.desc.shape[1]
    print(f"desc_dim: {desc_dim}")
    print(
        f"first molecule: x.shape={tuple(first.x.shape)}, "
        f"edge_index.shape={tuple(first.edge_index.shape)}, "
        f"y={first.y.item()}, split='{first.split}'"
    )

    assert desc_dim > 0, "desc_dim must be positive"
    assert first.desc.shape[0] == 1, "desc must be shape [1, desc_dim]"
    assert first.desc.isfinite().all(), \
        "first molecule has NaN/Inf in descriptor vector"
    assert first.x.shape[1] == 9, \
        "MoleculeNet-style featurisation = 9 atom features per atom"
    assert first.edge_index.shape[0] == 2, \
        "edge_index must be shape [2, num_directed_edges]"
