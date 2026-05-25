"""DGT training pipeline (`train.mode: dgt`).

Behaviour differs from upstream `custom_train` in three ways:

1. The **test set is not touched during training** — each epoch only runs
   train + val. The test loader is held out and used exactly once, after the
   training loop, on the best-val checkpoint.
2. Only the **best-val checkpoint** is kept on disk (`ckpt_best=True` +
   `ckpt_clean=True` enforced) — saves storage and makes "which model do I
   keep?" unambiguous.
3. After the final test pass, **per-sample test predictions are dumped** to
   `<run_dir>/test/predictions.pt` for post-hoc analysis (see
   `scripts/analyze_run.py`).

Registered as a parallel alternative to the upstream `custom` train mode (the
integration pattern documented in tech.md). Opt in via YAML:
    train:
      mode: dgt
"""
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch

from graphgps.train.custom_train import train_epoch


@torch.no_grad()
def _eval_and_collect(logger, loader, model, split):
    """Run inference on `loader`, update `logger`, and return (y_true, y_pred).

    Same per-batch flow as upstream `eval_epoch` but additionally collects all
    per-sample tensors so they can be saved for post-hoc analysis.
    """
    model.eval()
    ys_true, ys_pred = [], []
    time_start = time.time()
    for batch in loader:
        with torch.autocast(device_type="cuda"):
            batch.split = split
            batch.to(torch.device(cfg.device))
            pred, true = model(batch)
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(
            true=_true, pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=0, time_used=time.time() - time_start,
            params=cfg.params, dataset_name=cfg.dataset.name,
        )
        ys_true.append(_true)
        ys_pred.append(_pred)
        time_start = time.time()
    return torch.cat(ys_true, dim=0), torch.cat(ys_pred, dim=0)


@register_train('dgt')
def dgt_train(loggers, loaders, model, optimizer, scheduler):
    """DGT training: train+val loop, then a single test pass on the best ckpt."""
    if len(loaders) < 3 or len(loggers) < 3:
        raise ValueError("dgt train mode expects 3 loaders/loggers "
                         "(train, val, test).")

    # --- Enforce checkpoint settings needed for "best ckpt → final test" ---
    if not cfg.train.enable_ckpt:
        cfg.train.enable_ckpt = True
        logging.info("[dgt] forcing cfg.train.enable_ckpt = True")
    if not cfg.train.ckpt_best:
        cfg.train.ckpt_best = True
        logging.info("[dgt] forcing cfg.train.ckpt_best = True")
    if not cfg.train.ckpt_clean:
        cfg.train.ckpt_clean = True
        logging.info("[dgt] forcing cfg.train.ckpt_clean = True")

    train_logger, val_logger, test_logger = loggers
    train_loader, val_loader, test_loader = loaders

    perf_train, perf_val = [], []
    full_epoch_times = []
    best_epoch = 0

    m = cfg.metric_best
    agg = cfg.metric_agg  # 'argmax' for AUC, 'argmin' for MAE/MSE/RMSE/loss

    # --- Train + val loop (test loader untouched) ---
    for cur_epoch in range(cfg.optim.max_epoch):
        t0 = time.perf_counter()
        train_epoch(train_logger, train_loader, model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        perf_train.append(train_logger.write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            _eval_and_collect(val_logger, val_loader, model, split='val')
            perf_val.append(val_logger.write_epoch(cur_epoch))
        elif perf_val:
            perf_val.append(perf_val[-1])
        else:
            perf_val.append(perf_train[-1])  # before first eval epoch

        # Scheduler step.
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(perf_val[-1]['loss'])
        else:
            scheduler.step()

        # Determine best-val epoch and save its checkpoint (and clean older ones).
        if m != 'auto' and is_eval_epoch(cur_epoch):
            vals = np.array([vp[m] for vp in perf_val])
            best_epoch = int(getattr(vals, agg)())
            if best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                clean_ckpt()  # delete any older ckpts now that a new best exists

        full_epoch_times.append(time.perf_counter() - t0)
        logging.info(
            f"> Epoch {cur_epoch}: {full_epoch_times[-1]:.1f}s "
            f"(avg {np.mean(full_epoch_times):.1f}s) | "
            f"best so far: epoch {best_epoch} val_{m}: "
            f"{perf_val[best_epoch][m]:.4f}"
        )

    logging.info(f"[dgt] Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(
        f"[dgt] Total train loop time: "
        f"{np.sum(full_epoch_times) / 3600:.2f}h"
    )

    # --- Final test on the best-val checkpoint ---
    logging.info("[dgt] Loading best-val checkpoint for final test...")
    loaded_epoch = load_ckpt(model, optimizer, scheduler)
    logging.info(f"[dgt] Loaded checkpoint from epoch {loaded_epoch} "
                 f"(best by val_{m}).")

    y_true, y_pred = _eval_and_collect(test_logger, test_loader, model,
                                       split='test')
    test_stats = test_logger.write_epoch(best_epoch)
    logging.info(f"[dgt] Final test stats @ best-val epoch {best_epoch}: "
                 f"{test_stats}")

    # Dump per-sample predictions for post-hoc analysis.
    pred_path = Path(cfg.run_dir) / 'test' / 'predictions.pt'
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {'y_true': y_true, 'y_pred': y_pred,
         'best_epoch': best_epoch,
         'metric_best': m,
         'best_val_metric': float(perf_val[best_epoch][m])},
        pred_path,
    )
    logging.info(f"[dgt] Wrote test predictions: {pred_path}")

    for lg in loggers:
        lg.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()  # one more sweep just in case
    logging.info(f"[dgt] Task done, results saved in {cfg.run_dir}")
