"""Select descriptor columns from a SHAP-ranking CSV (for variant 5 of the
GWU descriptor-type study, documents/projects/gwu.md).

Reads a SHAP feature-ranking CSV **from S3** (via trans_learn's S3Handler, same
auth path as scripts/prepare_data.py), applies one selection criterion, and
prints a paste-ready ``desc_columns`` list + ``desc_dim`` for a
``*-WithDesc-*.yaml`` config.

The CSV is the output of the user's SHAP analysis pipeline; expected columns
(tab-separated): ``features``, ``mean_shap_abs`` (+ mean_shap, direction, ...).
``features`` names must match the dataset's descriptor column names (so the
loader's ``dataset.desc_columns`` can select them).

Known SHAP-ranking S3 keys (bucket = trans_learn PROJECT_BUCKET):
    GWU, QM only        ts_project_1/data/biodegradation/features/GWU/feature_rank_shap_gwu_b2_qm.csv
    GWU, RDKit only     ts_project_1/data/biodegradation/features/GWU/feature_rank_shap_gwu_b2_rdkit.csv
    GWU, QM + RDKit     ts_project_1/data/biodegradation/features/GWU/feature_rank_shap_gwu_b2_qm_rdkit.csv
    no-Reaxys, RDKit    ts_project_1/data/biodegradation/features/feature_rank_shap_no_reaxys_rdkit.csv

Criteria (exactly one; all rank by mean|SHAP|):
    --cumulative 0.90   keep the smallest top-set covering >=90% of total mean|SHAP|
                        (scale-free; PCA-explained-variance analog; recommended)
    --top-k 20          keep the top 20 features
    --abs 0.01          keep features with mean|SHAP| >= 0.01 (scale-dependent)
    --coef-thresh 0.10  keep features with mean|SHAP| >= 0.10 * max(mean|SHAP|)

Usage (run on the lab box where trans_learn + AWS creds live; needs boto3 /
python-dotenv in the active env):
    python scripts/select_features_from_shap.py \\
        --s3-key ts_project_1/data/biodegradation/features/GWU/feature_rank_shap_gwu_b2_qm.csv \\
        --trans-learn-path /home/jovyan/tools/trans_learn \\
        --cumulative 0.90

Output (printed to stdout)::

    Reading SHAP ranking from s3://<bucket>/.../feature_rank_shap_gwu_b2_qm.csv

    SHAP selection - cumulative=0.9: 24 features (covers 90.4% of total mean|SHAP|)  (of 40 ranked features)

    # paste into the variant config (loader re-orders to manifest order):
    dataset:
      desc_dim: 24
      desc_columns: ['weakly_polar_surface_area_gwu', 'count_atoms_gwu', ...]

How to use the output:
    1. Copy a base config, e.g. configs/biodegradability/Biodeg-GWU-DGT-Pipeline-WithDesc-gwu.yaml,
       to a variant-5 file (e.g. ...-WithDesc-sel.yaml).
    2. Under ``dataset:`` remove ``desc_include`` / ``desc_exclude`` and paste
       the printed ``desc_dim`` and ``desc_columns`` lines (keep
       ``standardize_desc: True``). ``desc_dim`` MUST equal len(desc_columns) —
       the head asserts it.
    3. Run training as usual:
         python main.py --cfg configs/biodegradability/...-WithDesc-sel.yaml \\
           --repeat 4 seed 0 wandb.use False optim.max_epoch 50
       The loader auto-keys a SEPARATE processed cache from the resolved column
       set (hash), so this subset never collides with other variants.
    Tip: redirect to capture just the YAML block, e.g.
         python scripts/select_features_from_shap.py ... --cumulative 0.90 | tail -n 3

Caveats (see gwu.md):
    - SHAP importance is MODEL-SPECIFIC: this ranking comes from the analysis
      model, not DGT, so selection is a screening heuristic, not DGT-faithful.
    - The ranking must have been computed on TRAIN data only (else leakage).
    - Selection order here is by importance; the loader re-orders to the
      manifest column order, so paste order does not matter.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_trans_learn(tl_path: Path) -> Path:
    """Return the sys.path entry that makes `trans_learn` importable.

    Mirrors scripts/prepare_data.py: handles src-layout, repo-root-with-package,
    and tl_path-is-the-package-dir.
    """
    tl_path = tl_path.resolve()
    if not tl_path.is_dir():
        raise FileNotFoundError(f"--trans-learn-path not found: {tl_path}")
    for c in (tl_path / 'src', tl_path, tl_path.parent):
        if (c / 'trans_learn' / 'settings.py').is_file():
            return c
    raise FileNotFoundError(
        f"Could not locate 'trans_learn/settings.py' relative to {tl_path}. "
        "Pass --trans-learn-path pointing at the dir whose child 'trans_learn/' "
        "(or 'src/trans_learn/') contains settings.py."
    )


def select(df: pd.DataFrame, *, cumulative=None, top_k=None,
           abs_thresh=None, coef_thresh=None):
    """Return (selected_feature_names, info_str) for the chosen criterion."""
    df = df.sort_values('mean_shap_abs', ascending=False).reset_index(drop=True)
    vals = df['mean_shap_abs'].astype(float)
    total = float(vals.sum())
    mx = float(vals.max())

    if cumulative is not None:
        frac = vals.cumsum() / total
        # smallest prefix reaching the target coverage
        k = int((frac < cumulative).sum()) + 1
        k = min(k, len(df))
        sel = df.iloc[:k]
        info = (f"cumulative={cumulative}: {k} features "
                f"(covers {float(frac.iloc[k-1])*100:.1f}% of total mean|SHAP|)")
    elif top_k is not None:
        sel = df.iloc[:top_k]
        info = f"top-k={top_k}: {len(sel)} features"
    elif abs_thresh is not None:
        sel = df[vals >= abs_thresh]
        info = f"abs>={abs_thresh}: {len(sel)} features"
    elif coef_thresh is not None:
        cut = coef_thresh * mx
        sel = df[vals >= cut]
        info = (f"coef-thresh={coef_thresh} (>= {cut:.4f} = {coef_thresh}*max): "
                f"{len(sel)} features")
    else:  # pragma: no cover - argparse guarantees one
        raise ValueError("no criterion given")

    names = sel['features'].tolist()
    if not names:
        raise ValueError(f"selection produced 0 features ({info}).")
    return names, info


def main():
    parser = argparse.ArgumentParser(
        description="Select descriptor columns from a SHAP-ranking CSV (S3).",
    )
    parser.add_argument(
        "--s3-key", required=True,
        help="S3 key of the SHAP-ranking CSV (relative to trans_learn "
             "PROJECT_BUCKET). See the module docstring for known keys.",
    )
    parser.add_argument(
        "--trans-learn-path", type=Path, required=True,
        help="Path to the trans_learn repo root (provides S3Handler + bucket).",
    )
    crit = parser.add_mutually_exclusive_group(required=True)
    crit.add_argument("--cumulative", type=float,
                      help="Keep the smallest top-set covering >= this fraction "
                           "(0-1) of total mean|SHAP| (recommended).")
    crit.add_argument("--top-k", type=int, help="Keep the top K features.")
    crit.add_argument("--abs", dest="abs_thresh", type=float,
                      help="Keep features with mean|SHAP| >= this absolute value.")
    crit.add_argument("--coef-thresh", type=float,
                      help="Keep features with mean|SHAP| >= coef_thresh * max.")
    args = parser.parse_args()

    if args.cumulative is not None and not (0 < args.cumulative <= 1):
        parser.error("--cumulative must be in (0, 1].")

    # Wire up trans_learn for S3Handler + PROJECT_BUCKET (deferred import: path
    # comes from CLI).
    tl_root = _resolve_trans_learn(args.trans_learn_path)
    sys.path.insert(0, str(tl_root))
    try:
        from trans_learn.settings import PROJECT_BUCKET
        from trans_learn.utils.awstools import S3Handler
    except Exception as e:
        raise RuntimeError(
            "Failed to import trans_learn (S3Handler / PROJECT_BUCKET). Likely: "
            "deps missing (boto3, python-dotenv) or trans_learn .env not found "
            f"from CWD. Underlying error: {e}"
        ) from e

    print(f"Reading SHAP ranking from s3://{PROJECT_BUCKET}/{args.s3_key}")
    s3 = S3Handler()
    with s3.open_file(args.s3_key) as fh:
        df = pd.read_csv(fh, sep='\t')

    for col in ('features', 'mean_shap_abs'):
        if col not in df.columns:
            raise KeyError(
                f"column '{col}' not in CSV; found {list(df.columns)}.")

    names, info = select(
        df, cumulative=args.cumulative, top_k=args.top_k,
        abs_thresh=args.abs_thresh, coef_thresh=args.coef_thresh,
    )

    print(f"\nSHAP selection — {info}  (of {len(df)} ranked features)\n")
    print("# paste into the variant config (loader re-orders to manifest order):")
    print("dataset:")
    print(f"  desc_dim: {len(names)}")
    print(f"  desc_columns: {names}")


if __name__ == "__main__":
    main()
