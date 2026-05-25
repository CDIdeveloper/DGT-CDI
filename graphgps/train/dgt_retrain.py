"""Retrain modes for final deployment models.

Two parallel train modes are registered here:

  - ``dgt_retrain``           — train on **train + val** combined. Test is
                                 held out. Default / methodologically clean.
  - ``dgt_retrain_with_test`` — train on **train + val + test** combined.
                                 Opt-in. The model sees test labels, so there
                                 is **no held-out data left** for any future
                                 evaluation.

Both are companions to ``train.mode: dgt``. They're invoked by
``scripts/retrain_on_trainval.py`` after the experiment phase to produce a
single deterministic model for downstream deployment.

The script picks the median-test seed from the original ``dgt``-mode runs and
its best-val epoch; this train mode then trains for ``cfg.optim.max_epoch``
epochs on the combined data and saves the resulting checkpoint.
"""
import logging
from pathlib import Path

import torch
from torch_geometric.graphgym.checkpoint import save_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train
from torch_geometric.loader import DataLoader

from graphgps.train.custom_train import train_epoch


def _run_retrain(loggers, loaders, model, optimizer, scheduler,
                 *, include_test: bool):
    """Shared retrain loop.

    Combines train + val (and optionally test) into a single training loader,
    trains for cfg.optim.max_epoch epochs, saves one final checkpoint.
    """
    if len(loaders) < 2:
        raise ValueError(
            "retrain mode needs at least 2 loaders (train, val).")
    if include_test and len(loaders) < 3:
        raise ValueError(
            "dgt_retrain_with_test needs 3 loaders (train, val, test).")

    train_logger = loggers[0]
    train_loader, val_loader = loaders[0], loaders[1]

    datasets = [train_loader.dataset, val_loader.dataset]
    if include_test:
        datasets.append(loaders[2].dataset)
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=getattr(train_loader, 'pin_memory', False),
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    tag = '[dgt_retrain_with_test]' if include_test else '[dgt_retrain]'

    if include_test:
        n_test = len(loaders[2].dataset)
        logging.warning(
            f"{tag} Combining train + val + TEST: "
            f"{n_train} + {n_val} + {n_test} = {n_train + n_val + n_test} samples."
        )
        logging.warning(
            f"{tag} Test labels are being used for training. The resulting "
            f"model has NO held-out data for any future evaluation; do not "
            f"re-test it on this dataset's test split (would be leakage)."
        )
    else:
        logging.info(
            f"{tag} Combining train + val: "
            f"{n_train} + {n_val} = {n_train + n_val} samples. "
            f"Test set is NOT touched."
        )
    logging.info(
        f"{tag} Training for {cfg.optim.max_epoch} epochs."
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
            f"{tag} Epoch {cur_epoch}: train_loss={stats['loss']:.4f}"
        )

    final_epoch = cfg.optim.max_epoch - 1
    save_ckpt(model, optimizer, scheduler, final_epoch)
    final_ckpt = Path(cfg.run_dir) / 'ckpt' / f'{final_epoch}.ckpt'
    logging.info(f"{tag} Saved final model: {final_ckpt}")
    train_logger.close()


@register_train('dgt_retrain')
def dgt_retrain(loggers, loaders, model, optimizer, scheduler):
    """Retrain on train + val combined. Test loader is NEVER read."""
    _run_retrain(loggers, loaders, model, optimizer, scheduler,
                 include_test=False)


@register_train('dgt_retrain_with_test')
def dgt_retrain_with_test(loggers, loaders, model, optimizer, scheduler):
    """Retrain on train + val + TEST combined. Opt-in; no held-out data left."""
    _run_retrain(loggers, loaders, model, optimizer, scheduler,
                 include_test=True)
