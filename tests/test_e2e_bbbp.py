"""End-to-end regression test for the DGT training pipeline.

Runs a *reduced* BBBP training run (2 seeds x 30 epochs) via ``main.py`` and
checks that the pipeline trains cleanly and clears a conservative
validation-AUC floor.

Purpose: guard the core DGT pipeline against regressions introduced while
building the biodegradability dataset loader and descriptor head (Roadmap
Phases 1-3 in ``documents/overview.md``). This is a regression *gate*, not a
paper reproduction -- 30 epochs is deliberately under-trained, so the AUC
floor is set well below the paper's BBBP result.

Marked ``e2e``: excluded from the default ``pytest`` run; invoke explicitly
with ``pytest -m e2e``. Requires CUDA (``main.py`` hardcodes ``cuda:0``).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG = "configs/physiology/BBBP-RWSE-SPDE-Rings.yaml"
CONFIG_NAME = "BBBP-RWSE-SPDE-Rings"

# Reduced-run settings -- see Roadmap Phase 0 in documents/overview.md.
N_SEEDS = 2
MAX_EPOCH = 30
VAL_AUC_FLOOR = 0.62   # conservative; tighten once real 30-epoch numbers are seen
RUN_TIMEOUT_S = 3600


def _read_stats_jsonl(path):
    """Per-epoch stats are appended one JSON object per line by CustomLogger."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.mark.e2e
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DGT pipeline requires CUDA (main.py hardcodes cuda:0)",
)
def test_bbbp_reduced_run(tmp_path):
    cmd = [
        sys.executable, "main.py",
        "--cfg", CONFIG,
        "--repeat", str(N_SEEDS),
        "seed", "0",
        "wandb.use", "False",
        "optim.max_epoch", str(MAX_EPOCH),
        "out_dir", str(tmp_path),
    ]
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=RUN_TIMEOUT_S,
    )

    assert result.returncode == 0, (
        f"main.py exited with code {result.returncode}\n"
        f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
    )

    run_root = tmp_path / CONFIG_NAME
    assert run_root.is_dir(), f"run directory {run_root} was not created"

    for seed in range(N_SEEDS):
        seed_dir = run_root / str(seed)

        for split in ("train", "val", "test"):
            stats_path = seed_dir / split / "stats.json"
            assert stats_path.is_file(), f"missing {stats_path}"
            stats = _read_stats_jsonl(stats_path)
            assert len(stats) == MAX_EPOCH, (
                f"seed {seed} / {split}: expected {MAX_EPOCH} epochs of stats, "
                f"got {len(stats)}"
            )
            for epoch_stats in stats:
                assert not math.isnan(epoch_stats["loss"]), (
                    f"seed {seed} / {split}: NaN loss at epoch "
                    f"{epoch_stats['epoch']}"
                )

        val_stats = _read_stats_jsonl(seed_dir / "val" / "stats.json")
        best_val_auc = max(s["auc"] for s in val_stats)
        assert best_val_auc > VAL_AUC_FLOOR, (
            f"seed {seed}: best validation AUC {best_val_auc:.4f} is below "
            f"the floor {VAL_AUC_FLOOR} -- the pipeline may be broken"
        )
