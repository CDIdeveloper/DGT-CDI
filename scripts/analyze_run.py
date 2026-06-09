"""Post-hoc analysis of a DGT run.

Usage:
    python scripts/analyze_run.py <run_dir> [--no-plots]

`run_dir` is a per-seed directory, e.g.
    results/DGT/BBBP-DGT-Pipeline/0

Expects:
    <run_dir>/config.yaml                  (auto-dumped by GraphGym)
    <run_dir>/test/predictions.pt          (written by train.mode: dgt)

Produces a summary in <run_dir>/plots/summary.json plus, unless --no-plots,
the plot set for the task type:
    classification_binary  → ROC, PR, confusion matrix @ optimal-F1, score histogram
    regression             → actual-vs-predicted scatter, residuals, residual histogram

`--no-plots` computes and writes summary.json only (skips the PNGs) — useful
when you just need the metrics / best_f1_threshold. summary.json stays at
<run_dir>/plots/summary.json either way, so retrain_on_trainval.py can still
find best_f1_threshold.

The metric + plot logic lives in the shared _eval_plots module (also used by
predict.py). CPU-only; matplotlib + scikit-learn.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

# Make the sibling _eval_plots module importable regardless of invocation/cwd.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _eval_plots import analyze_classification_binary, analyze_regression  # noqa: E402


def _load_config(run_dir: Path) -> dict:
    """Load the dumped GraphGym config.

    GraphGym writes a single `config.yaml` to the parent `out_dir` (shared by
    all `--repeat N` seed subdirs), but some workflows also keep a copy inside
    the per-seed dir. Search both — per-seed first, then parent.
    """
    for candidate in (run_dir / "config.yaml", run_dir.parent / "config.yaml"):
        if candidate.is_file():
            with open(candidate) as fh:
                return yaml.safe_load(fh)
    raise FileNotFoundError(
        f"config.yaml not found in {run_dir} or {run_dir.parent}. "
        "GraphGym normally writes it to the parent (out_dir). Confirm "
        f"{run_dir} is a per-seed directory like results/DGT/<config_name>/<seed>."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis of a single-seed DGT run "
                    "(summary.json + plots under <run_dir>/plots/)."
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Per-seed run directory (e.g. results/DGT/BBBP-DGT-Pipeline/0).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Compute and write summary.json only; skip the PNG plots.",
    )
    args = parser.parse_args()
    make_plots = not args.no_plots

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")

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
        summary = analyze_classification_binary(y_true, y_pred, plot_dir,
                                                make_plots=make_plots)
    elif task_type == "regression":
        summary = analyze_regression(y_true, y_pred, plot_dir,
                                     make_plots=make_plots)
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

    where = "summary only (no plots)" if args.no_plots else "plots + summary"
    print(f"Analysis complete ({where}): {plot_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
