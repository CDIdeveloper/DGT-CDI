"""Manuscript figures for the biodegradability DGT study.

Two figures, both built from artifacts already on disk. Nothing is trained and no
metric is recomputed in a way that could disagree with the paper — each figure
plots the same quantity its section reports.

  fig1_paired_fold_auc    the §7 paired descriptor test, made visual.
                          (a) per-fold OOF ROC-AUC for the three exported arms,
                          one line per fold, so the pairing is on the page: fold
                          identity moves the metric far more than arm identity
                          does. (b) the two paired contrasts as per-fold
                          differences against zero — the quantity the t-tests in
                          §7 actually test.
                          Source: results/predictions/dgt_oof_test_*.parquet
                          (train rows = out-of-fold, one model per fold).

  fig2_threshold_plateau  the §7 threshold-identifiability finding. F1,
                          precision and recall against decision threshold for
                          both deployed models, with the "within 0.01 of max F1"
                          plateau shaded. F1 is flat across roughly half the
                          probability scale while precision and recall vary
                          smoothly — so the F1 argmax is noise and 0.5 costs
                          almost nothing.
                          Source: the deployment bundles' deploy_eval/*_scores.csv
                          — the same models §7 quotes. NOT the ablation seeds.

Thresholds are swept over the distinct predicted scores, which is what
`precision_recall_curve` and `_eval_plots.best_f1_threshold` do, so the plateau
endpoints reproduce §7's numbers rather than approximating them.

Usage (lab box, repo root, `dgt` env — CPU only, needs matplotlib):
    python scripts/make_paper_figures.py

Writes PDF (vector, for the manuscript) and PNG (300 dpi, for preview) into
results/figures/, which is gitignored.

Design notes: static print figures, so there is no hover layer and no dark mode.
The categorical slots used (blue/orange/aqua) were validated for all-pairs CVD
and normal-vision separation; aqua falls below 3:1 against the surface, so every
curve carries a visible direct label rather than relying on colour.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = REPO_ROOT / 'results' / 'predictions'
FINAL_MODELS = REPO_ROOT / 'results' / 'final_models'
OUT_DIR = REPO_ROOT / 'results' / 'figures'

# Arms in order of descriptor content, which is the axis order in fig 1a.
ARMS = [
    ('none', 'graph only'),
    ('rdkit_fg', 'RDKit/fg\n(207)'),
    ('qm_rdkit', 'all\n(247)'),
]
# The two contrasts §7 reports, as (label, minuend, subtrahend).
CONTRASTS = [
    ('RDKit/fg − graph only', 'rdkit_fg', 'none'),
    ('all − RDKit/fg', 'qm_rdkit', 'rdkit_fg'),
]
# Deployment bundles for fig 2, with the score-CSV basename each one wrote.
BUNDLES = [
    ('biodeg-no-ind-dgt-nongwu-2026-09-03', 'nongwu_v2_scores.csv',
     'Descriptor model (RDKit/fg, 207)'),
    ('biodeg-no-ind-dgt-graphonly-2026-09-03', 'graphonly_v2_scores.csv',
     'Graph-only model'),
]
SCORE_COL, LABEL_COL = 'y_pred_score', 'degradable'
PLATEAU_TOL = 0.01  # "within 0.01 of the maximum" (§7)

# Validated categorical slots 1-3 (references/palette.md), plus ink tokens.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK_2, INK_MUTED = '#0b0b0b', '#52514e', '#8a8985'


def _style():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.edgecolor': INK_MUTED, 'axes.linewidth': 0.6,
        'axes.labelcolor': INK_2, 'text.color': INK,
        'xtick.color': INK_2, 'ytick.color': INK_2,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': '#e8e7e3', 'grid.linewidth': 0.6,
        'legend.frameon': False, 'legend.fontsize': 7.5,
        'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    })


def _save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 300})):
        path = OUT_DIR / f'{stem}.{ext}'
        fig.savefig(path, **kw)
        print(f'  wrote {path}')
    plt.close(fig)


def _prf_sweep(y_true, score):
    """Precision, recall and F1 over the PR-curve thresholds, using `score >= t`.

    Deliberately the same threshold set and the same comparison as
    `scripts/_eval_plots.best_f1_threshold`, which is the code path §7's plateau
    numbers came from — so the endpoints reproduce those numbers rather than
    approximating them. Threshold 0 is prepended so the curve spans the axis.
    """
    _, _, thresholds = precision_recall_curve(y_true, score)
    thresholds = np.concatenate([[0.0], np.asarray(thresholds, dtype=float)])
    precision, recall, f1 = [], [], []
    for t in thresholds:
        pred = (score >= t).astype(int)
        precision.append(precision_score(y_true, pred, zero_division=0))
        recall.append(recall_score(y_true, pred, zero_division=0))
        f1.append(f1_score(y_true, pred, zero_division=0))
    return thresholds, np.array(precision), np.array(recall), np.array(f1)


def _fold_aucs():
    """{arm: [auc per fold]} from the out-of-fold train rows of each export."""
    out = {}
    for arm, _ in ARMS:
        path = PRED_DIR / f'dgt_oof_test_{arm}.parquet'
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/export_oof_test_predictions.py "
                f"--arm {arm} first."
            )
        df = pd.read_parquet(path, columns=['split', 'true', 'prob', 'fold'])
        tr = df[df['split'] == 'train']
        out[arm] = [
            roc_auc_score(g['true'].to_numpy(), g['prob'].to_numpy())
            for _, g in tr.groupby('fold')
        ]
        print(f'  {arm:<9} per-fold ROC-AUC ' +
              ', '.join(f'{v:.4f}' for v in out[arm]) +
              f'   mean {np.mean(out[arm]):.4f} ± {np.std(out[arm]):.4f}')
    return out


def figure_1():
    print('Figure 1 — paired per-fold OOF ROC-AUC')
    aucs = _fold_aucs()
    n_folds = len(next(iter(aucs.values())))
    xs = np.arange(len(ARMS))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), gridspec_kw={'width_ratios': [1.15, 1]})

    # (a) one line per fold + the across-fold mean.
    ax_a.set_axisbelow(True)
    ax_a.yaxis.grid(True)
    for f in range(n_folds):
        ax_a.plot(xs, [aucs[a][f] for a, _ in ARMS], color=INK_MUTED,
                  lw=1.0, marker='o', ms=3.5, alpha=0.75, zorder=2)
    means = [float(np.mean(aucs[a])) for a, _ in ARMS]
    ax_a.plot(xs, means, color=BLUE, lw=2.0, marker='o', ms=6,
              markeredgecolor='white', markeredgewidth=1.2, zorder=3)
    ax_a.annotate('mean of 5 folds', (xs[-1], means[-1]),
                  textcoords='offset points', xytext=(-4, 10),
                  ha='right', color=BLUE, fontsize=7.5, fontweight='bold')
    ax_a.annotate('individual folds', (xs[0], min(aucs[ARMS[0][0]])),
                  textcoords='offset points', xytext=(6, -12),
                  ha='left', color=INK_2, fontsize=7.5)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([lbl for _, lbl in ARMS])
    ax_a.set_xlim(-0.35, len(ARMS) - 0.65)
    ax_a.set_ylabel('Out-of-fold ROC-AUC')
    ax_a.set_title('(a) Every fold, every arm', loc='left', color=INK)

    # (b) paired differences against zero — what §7's t-tests test.
    ax_b.set_axisbelow(True)
    ax_b.yaxis.grid(True)
    ax_b.axhline(0, color=INK_2, lw=0.9, zorder=2)
    for i, (label, hi, lo) in enumerate(CONTRASTS):
        diffs = np.array(aucs[hi]) - np.array(aucs[lo])
        colour = (BLUE, ORANGE)[i]
        ax_b.plot(np.full(len(diffs), i), diffs, marker='o', ms=5, lw=0,
                  color=colour, markeredgecolor='white', markeredgewidth=1.0,
                  alpha=0.95, zorder=3)
        ax_b.plot([i - 0.16, i + 0.16], [diffs.mean()] * 2, color=colour,
                  lw=2.0, zorder=4)
        favour = int((diffs > 0).sum())
        ax_b.annotate(f'mean {diffs.mean():+.4f}\n{favour}/{len(diffs)} folds > 0',
                      (i + 0.2, diffs.mean()), va='center', ha='left',
                      color=colour, fontsize=7.5)
        print(f'  {label:<22} mean {diffs.mean():+.4f}  '
              f'{favour}/{len(diffs)} folds > 0')
    ax_b.set_xticks(range(len(CONTRASTS)))
    ax_b.set_xticklabels([lbl for lbl, _, _ in CONTRASTS])
    ax_b.set_xlim(-0.45, len(CONTRASTS) + 0.45)
    ax_b.set_ylabel('Δ ROC-AUC, paired by fold')
    ax_b.set_title('(b) Paired differences', loc='left', color=INK)

    fig.tight_layout()
    _save(fig, 'fig1_paired_fold_auc')


def figure_2():
    print('Figure 2 — F1 threshold plateau')
    panels = []
    for bundle, csv_name, title in BUNDLES:
        path = FINAL_MODELS / bundle / 'deploy_eval' / csv_name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. §7's plateau is measured on the DEPLOYED "
                f"models, so this figure needs the bundle's deploy_eval CSV."
            )
        df = pd.read_csv(path)
        for col in (SCORE_COL, LABEL_COL):
            if col not in df.columns:
                raise KeyError(
                    f"{path} has no '{col}' column (found {list(df.columns)[:8]}"
                    f"...). Tell me the real column names and I will adjust."
                )
        keep = df[[SCORE_COL, LABEL_COL]].apply(pd.to_numeric, errors='coerce').dropna()
        panels.append((title, keep[LABEL_COL].to_numpy(),
                       keep[SCORE_COL].to_numpy()))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for ax, (title, y_true, score) in zip(axes, panels):
        thr, precision, recall, f1 = _prf_sweep(y_true, score)
        best = f1.max()
        inside = thr[f1 >= best - PLATEAU_TOL]
        lo, hi = float(inside.min()), float(inside.max())
        argmax_t = float(thr[int(np.argmax(f1))])
        f1_at_half = float(f1[np.searchsorted(thr, 0.5, side='right') - 1])

        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.axvspan(lo, hi, color=BLUE, alpha=0.09, lw=0, zorder=1)
        for series, colour, label in ((recall, AQUA, 'Recall'),
                                      (precision, ORANGE, 'Precision'),
                                      (f1, BLUE, 'F1')):
            ax.plot(thr, series, color=colour, lw=2.0 if label == 'F1' else 1.4,
                    drawstyle='steps-post', zorder=3,
                    alpha=1.0 if label == 'F1' else 0.9)
            ax.annotate(label, (0.985, series[np.searchsorted(thr, 0.985) - 1]),
                        textcoords='offset points', xytext=(3, 0), va='center',
                        ha='left', color=colour, fontsize=7.5,
                        fontweight='bold' if label == 'F1' else 'normal',
                        annotation_clip=False)
        ax.axvline(0.5, color=INK_2, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.annotate('0.5', (0.5, 1.005), ha='center', va='bottom',
                    color=INK_2, fontsize=7.5, annotation_clip=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Decision threshold')
        ax.set_title(title, loc='left', color=INK)
        ax.annotate(
            f'F1 within {PLATEAU_TOL:g} of max\nover {lo:.3f}–{hi:.3f} '
            f'(width {hi - lo:.3f})',
            (0.03, 0.06), xycoords='axes fraction', ha='left', va='bottom',
            color=BLUE, fontsize=7)
        print(f'  {title}')
        print(f'    max F1 {best:.4f} at threshold {argmax_t:.4f};  '
              f'F1 @ 0.5 = {f1_at_half:.4f}  (cost {best - f1_at_half:.4f})')
        print(f'    plateau {lo:.4f}–{hi:.4f}  width {hi - lo:.4f}  '
              f'contiguous={len(inside) == int((f1 >= best - PLATEAU_TOL).sum())}')
    axes[0].set_ylabel('Metric value')
    fig.tight_layout()
    _save(fig, 'fig2_threshold_plateau')


def main():
    _style()
    figure_1()
    print()
    figure_2()
    print(f'\nFigures in {OUT_DIR}. Upload the PDFs with:\n'
          f'  aws s3 cp {OUT_DIR}/ '
          f's3://cdi-lab-workspaces/ts_project_1/data/biodegradation/GWU/'
          f'paper_artifacts/dgt/figures/ --recursive --exclude "*" '
          f'--include "*.pdf" --include "*.png"')


if __name__ == '__main__':
    main()
