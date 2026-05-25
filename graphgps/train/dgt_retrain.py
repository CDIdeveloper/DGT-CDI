"""Retrain on combined train+val (`train.mode: dgt_retrain`).

Companion to `train.mode: dgt`. Used by `scripts/retrain_on_trainval.py` to
produce a single deterministic model for downstream deployment, *after* the
experiment phase is done.

Workflow:
  1. The user runs the experiment with `train.mode: dgt` (--repeat N). Each
     seed produces its own best-val checkpoint and a final test record.
  2. `scripts/retrain_on_trainval.py` picks the seed whose test metric is
     closest to the median across seeds and reads that seed's best-val epoch.
  3. It then invokes main.py with `train.mode: dgt_retrain` and overrides for
     `seed` and `optim.max_epoch`.
  4. This train mode trains on **train + val combined** (NO test data) for
     `cfg.optim.max_epoch` epochs and saves the resulting checkpoint.

The test set (`loaders[2]`) is NEVER touched in this mode. There is no
validation phase either — by construction, val is folded into the training
set. Use only AFTER you've already estimated generalisation via per-seed
test metrics from the original dgt-mode runs.
"""
import logging
from pathlib import Path

import torch
from torch_geometric.graphgym.checkpoint import save_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train
from torch_geometric.loader import DataLoader

from graphgps.train.custom_train import train_epoch


@register_train('dgt_retrain')
def dgt_retrain(loggers, loaders, model, optimizer, scheduler):
    """Combine train+val, train for cfg.optim.max_epoch, save the final ckpt."""
    if len(loaders) < 2:
        raise ValueError(
            "dgt_retrain needs at least 2 loaders (train, val).")

    train_logger = loggers[0]
    train_loader, val_loader = loaders[0], loaders[1]
    # loaders[2] (test) is held out — NEVER read in this mode.

    # Combine train + val datasets into a single training set.
    combined_dataset = torch.utils.data.ConcatDataset(
        [train_loader.dataset, val_loader.dataset]
    )
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=getattr(train_loader, 'pin_memory', False),
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    logging.info(
        f"[dgt_retrain] Combined train+val: "
        f"{n_train} + {n_val} = {n_train + n_val} samples."
    )
    logging.info(
        f"[dgt_retrain] Training for {cfg.optim.max_epoch} epochs. "
        f"Test set is NOT touched."
    )

    for cur_epoch in range(cfg.optim.max_epoch):
        train_epoch(
            train_logger, combined_loader, model, optimizer, scheduler,
            cfg.optim.batch_accumulation,
        )
        stats = train_logger.write_epoch(cur_epoch)
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(stats['loss'])
        else:
            scheduler.step()
        logging.info(
            f"[dgt_retrain] Epoch {cur_epoch}: "
            f"train_loss={stats['loss']:.4f}"
        )

    # Save the final checkpoint (only one, at the last epoch).
    final_epoch = cfg.optim.max_epoch - 1
    save_ckpt(model, optimizer, scheduler, final_epoch)
    final_ckpt = Path(cfg.run_dir) / 'ckpt' / f'{final_epoch}.ckpt'
    logging.info(f"[dgt_retrain] Saved final model: {final_ckpt}")
    train_logger.close()
