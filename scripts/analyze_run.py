"""Post-hoc analysis of a DGT run.

Usage:
    python scripts/analyze_run.py <run_dir>

`run_dir` is a per-seed directory, e.g.
    results/DGT/BBBP-DGT-Pipeline/0

Expects:
    <run_dir>/config.yaml                  (auto-dumped by GraphGym)
    <run_dir>/test/predictions.pt          (written by train.mode: dgt)

Produces plots and a summary in <run_dir>/plots/.

Branches on cfg.dataset.task_type:
    classification_binary  → ROC, PR, confusion matrix @ optimal-F1, score histogram
    regression             → actual-vs-predicted scatter, residuals, residual histogram

CPU-only; matplotlib + scikit-learn.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe over SSH / headless

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _load_config(run_dir: Path) -> dict:
    with open(run_dir / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _to_1d_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().reshape(-1)


def _analyze_classification_binary(y_true_t, y_pred_t, plot_dir: Path) -> dict:
    y_true = _to_1d_numpy(y_true_t).astype(int)
    y_score = _to_1d_numpy(y_pred_t).astype(float)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "roc.png", dpi=150)
    plt.close(fig)

    # Precision-Recall
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test set)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(plot_dir / "pr.png", dpi=150)
    plt.close(fig)

    # Confusion matrix at optimal-F1 threshold (sweep PR thresholds).
    f1s = []
    for t in pr_thresh:
        y_hat = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, y_hat, zero_division=0))
    if len(f1s) == 0:
        best_t, best_f1 = 0.5, 0.0
    else:
        best_idx = int(np.argmax(f1s))
        best_t = float(pr_thresh[best_idx])
        best_f1 = float(f1s[best_idx])
    y_hat_best = (y_score >= best_t).astype(int)
    cm = confusion_matrix(y_true, y_hat_best, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix @ thr={best_t:.3f}  (F1={best_f1:.3f})")
    fig.tight_layout()
    fig.savefig(plot_dir / "confusion.png", dpi=150)
    plt.close(fig)

    # Score histogram per true class (visual proxy for calibration).
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_score[y_true == 0], bins=30, alpha=0.6, label="True 0")
    ax.hist(y_score[y_true == 1], bins=30, alpha=0.6, label="True 1")
    ax.set_xlabel("Predicted score (class 1)")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by true class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "score_hist.png", dpi=150)
    plt.close(fig)

    return {
        "task_type": "classification_binary",
        "n_samples": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "best_f1_threshold": best_t,
        "best_f1": best_f1,
        "confusion_matrix": cm.tolist(),
    }


def _analyze_regression(y_true_t, y_pred_t, plot_dir: Path) -> dict:
    y_true = _to_1d_numpy(y_true_t)
    y_pred = _to_1d_numpy(y_pred_t)
    residual = y_pred - y_true

    # Actual vs predicted scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="gray")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (test set)")
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter.png", dpi=150)
    plt.close(fig)

    # Residuals vs predicted
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_pred, residual, alpha=0.5, s=10)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (pred − actual)")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(plot_dir / "residual.png", dpi=150)
    plt.close(fig)

    # Residual histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residual, bins=30)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_hist.png", dpi=150)
    plt.close(fig)

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return {
        "task_type": "regression",
        "n_samples": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "mean_residual": float(np.mean(residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis of a single-seed DGT run "
                    "(plots + summary.json under <run_dir>/plots/)."
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Per-seed run directory (e.g. results/DGT/BBBP-DGT-Pipeline/0).",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Missing {cfg_path}. Make sure {run_dir} is a per-seed run "
            "directory (not the parent), and that GraphGym dumped the config."
        )
    cfg = _load_config(run_dir)
    task_type = cfg["dataset"]["task_type"]

    pred_path = run_dir / "test" / "predictions.pt"
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"Missing {pred_path}. Did you train with `train.mode: dgt`? "
            "The upstream `custom` train mode does not dump per-sample "
            "predictions."
        )
    pred = torch.load(pred_path, map_location="cpu", weights_only=False)
    y_true = pred["y_true"]
    y_pred = pred["y_pred"]
    best_epoch = pred.get("best_epoch")
    best_val_metric = pred.get("best_val_metric")
    metric_best = pred.get("metric_best")

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    if task_type == "classification_binary":
        summary = _analyze_classification_binary(y_true, y_pred, plot_dir)
    elif task_type == "regression":
        summary = _analyze_regression(y_true, y_pred, plot_dir)
    else:
        raise NotImplementedError(
            f"task_type={task_type!r} not yet supported by analyze_run.py"
        )

    summary["best_epoch"] = best_epoch
    summary["metric_best"] = metric_best
    summary["best_val_metric"] = best_val_metric
    summary["run_dir"] = str(run_dir)

    with open(plot_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Analysis complete: {plot_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
