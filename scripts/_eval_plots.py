"""Shared metric + plot helpers for DGT evaluation.

Single source of truth for the evaluation logic used by:
  - [analyze_run.py](analyze_run.py): post-hoc analysis of a training run's
    held-out test predictions (`<run_dir>/test/predictions.pt`).
  - [predict.py](predict.py): evaluation on a labelled input CSV (when
    `--label-col` is given).

Each analysis function **always computes the metrics** (and returns them as a
summary dict); plotting is optional via ``make_plots``. This keeps "metrics
only" (e.g. `analyze_run.py --no-plots`) and "metrics + plots" on one code
path, so the two scripts can never drift.

CPU-only; matplotlib (Agg backend) + scikit-learn.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe over SSH / headless

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def to_1d(arr) -> np.ndarray:
    """Flatten a torch tensor or array-like to a 1-D numpy array."""
    if hasattr(arr, "detach"):  # torch.Tensor
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr).reshape(-1)


def best_f1_threshold(y_true_t, y_pred_t):
    """Return (threshold, f1) maximising F1 over the PR-curve thresholds.

    Sweeping thresholds and keeping the maximum **fits a parameter to the split
    it is computed on**, so the resulting F1 is optimistically biased for that
    split. Derive the threshold from VALIDATION predictions and apply it to
    test; never sweep on test and then quote F1 at that threshold on the same
    test set (dgt_porting_guide.md §7 item 3 lists the decision threshold among
    the quantities that must be fit on train/val only).
    """
    y_true = to_1d(y_true_t).astype(int)
    y_score = to_1d(y_pred_t).astype(float)
    _, _, pr_thresh = precision_recall_curve(y_true, y_score)
    if len(pr_thresh) == 0:
        return 0.5, 0.0
    f1s = [f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
           for t in pr_thresh]
    best_idx = int(np.argmax(f1s))
    return float(pr_thresh[best_idx]), float(f1s[best_idx])


def analyze_classification_binary(y_true_t, y_pred_t, plot_dir,
                                  make_plots=True, f1_threshold=None) -> dict:
    """Binary-classification metrics (+ optional plots). Returns a summary dict.

    Metrics (ROC-AUC, average precision, optimal-F1 threshold, confusion
    matrix) are always computed. When ``make_plots`` is True, also writes
    roc.png / pr.png / confusion.png / score_hist.png into ``plot_dir``.

    Args:
        f1_threshold: decision threshold to report the confusion matrix and
            thresholded F1 at. Pass the value derived from VALIDATION
            predictions so the reported figures are not threshold-fitted to
            this split. When None, falls back to sweeping this split
            in-sample — convenient, but optimistically biased; the summary
            records which happened under ``f1_threshold_source``.
    """
    y_true = to_1d(y_true_t).astype(int)
    y_score = to_1d(y_pred_t).astype(float)

    roc_auc = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Always compute the in-sample sweep, for reference and for the fallback.
    insample_t, insample_f1 = best_f1_threshold(y_true, y_score)
    if f1_threshold is None:
        best_t, source = insample_t, "in-sample (this split)"
    else:
        best_t, source = float(f1_threshold), "validation"
    y_hat_best = (y_score >= best_t).astype(int)
    best_f1 = float(f1_score(y_true, y_hat_best, zero_division=0))
    cm = confusion_matrix(y_true, y_hat_best, labels=[0, 1])

    if make_plots:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(plot_dir / "roc.png", dpi=150)
        plt.close(fig)

        # Precision-Recall
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(rec, prec, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(plot_dir / "pr.png", dpi=150)
        plt.close(fig)

        # Confusion matrix at optimal-F1 threshold
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
        "f1_threshold_source": source,
        "best_f1": best_f1,
        # What an in-sample sweep on THIS split would have given. When
        # f1_threshold_source is "validation", the gap between best_f1 and
        # best_f1_insample is the optimism an in-sample threshold would buy.
        "best_f1_threshold_insample": insample_t,
        "best_f1_insample": insample_f1,
        "confusion_matrix": cm.tolist(),
    }


def analyze_regression(y_true_t, y_pred_t, plot_dir, make_plots=True) -> dict:
    """Single-target regression metrics (+ optional plots). Returns a summary.

    Metrics (MAE, RMSE, mean residual) are always computed. When
    ``make_plots`` is True, also writes scatter.png / residual.png /
    residual_hist.png into ``plot_dir``.
    """
    y_true = to_1d(y_true_t)
    y_pred = to_1d(y_pred_t)
    residual = y_pred - y_true

    if make_plots:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Actual vs predicted scatter
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="gray")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
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
