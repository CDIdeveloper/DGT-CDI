"""
Emit per-molecule MPNN predictions keyed by SMILES for one feature set:
  - OOF-on-train : 5-fold out-of-fold probabilities (each molecule scored by a fold model that did NOT
                   fit it) -- the same 5-fold procedure run_cv.py uses, but collected per molecule.
  - 278-test     : the final model (trained on all train, inner val for monitoring) scored once.

This is a generic, reusable exporter: the resulting table (smiles, split, true, prob) can be sliced by
partition bucket (Stage 4 (d)), used for calibration curves, error analysis, or joined to any other
model's per-molecule predictions for a cross-model comparison. It reuses the leak-free kfold_cv building
blocks with NO edits to run_cv.py / run_final.py, so the probabilities are consistent with the reported
MPNN CV / final numbers (default architecture = the HPO winner).

Output: data/pred_data/gwu_paper/mpnn_oof_test_<feature_set>.parquet

env (lab, GPU):
    cd src/trans_learn/models/m_chemprop/classification/kfold_cv/
    conda activate chemprop
    python predict_oof.py                        # feature set qm_rdkit (headline)
    python predict_oof.py --feature-set rdkit_fg
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from trans_learn.models.m_chemprop.classification.kfold_cv.cv_config import (
    FEATURE_SETS, K_FOLDS, VAL_FRACTION, RANDOM_STATE, SMILES_COL, PATH_PRED_RES, DEFAULT_MPNN_PARAMS,
)
from trans_learn.models.m_chemprop.classification.kfold_cv.mpnn_common import (
    load_split, build_datapoints, make_loaders_and_scaler, make_test_loader,
    build_and_fit, predict_probs,
)


def oof_train(feature_set, params):
    """5-fold OOF probabilities on the TRAIN split (same folds as run_cv), per molecule."""
    smis, ys, mol_descs = load_split('train', feature_set)
    datapoints = build_datapoints(smis, ys, mol_descs)
    smis, ys = np.asarray(smis), np.asarray(ys)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(ys), dtype=float)
    for fold_i, (trainval_idx, heldout_idx) in enumerate(skf.split(np.zeros(len(ys)), ys)):
        tr_idx, val_idx = train_test_split(
            trainval_idx, test_size=VAL_FRACTION, stratify=ys[trainval_idx], random_state=RANDOM_STATE,
        )
        train_dp = [datapoints[i] for i in tr_idx]
        val_dp = [datapoints[i] for i in val_idx]
        held_dp = [datapoints[i] for i in heldout_idx]
        loaders, scaler, input_dim = make_loaders_and_scaler(
            train_dp, val_dp, held_dp, params['hidden_dim_mp'])
        train_loader, val_loader, held_loader = loaders
        mpnn, trainer = build_and_fit(scaler, input_dim, train_loader, val_loader, params)
        oof[heldout_idx] = predict_probs(trainer, mpnn, held_loader)
        print(f"  fold {fold_i}: {len(heldout_idx)} held-out molecules scored")
    return pd.DataFrame({SMILES_COL: smis, 'split': 'train', 'true': ys, 'prob': oof})


def final_test(feature_set, params):
    """Final model (train on ALL train, inner val for monitoring) -> per-molecule 278-test probs."""
    smis, ys, mol_descs = load_split('train', feature_set)
    datapoints = build_datapoints(smis, ys, mol_descs)
    ys = np.asarray(ys)
    tr_idx, val_idx = train_test_split(
        np.arange(len(ys)), test_size=VAL_FRACTION, stratify=ys, random_state=RANDOM_STATE,
    )
    train_dp = [datapoints[i] for i in tr_idx]
    val_dp = [datapoints[i] for i in val_idx]
    loaders, scaler, input_dim = make_loaders_and_scaler(
        train_dp, val_dp, val_dp, params['hidden_dim_mp'])
    train_loader, val_loader, _ = loaders
    mpnn, trainer = build_and_fit(scaler, input_dim, train_loader, val_loader, params)

    test_smis, test_ys, test_descs = load_split('test', feature_set)
    test_dp = build_datapoints(test_smis, test_ys, test_descs)
    probs = predict_probs(trainer, mpnn, make_test_loader(test_dp))
    return pd.DataFrame({SMILES_COL: np.asarray(test_smis), 'split': 'test',
                         'true': np.asarray(test_ys), 'prob': probs})


def main():
    parser = argparse.ArgumentParser(description="Export per-molecule MPNN OOF-train + 278-test probabilities.")
    parser.add_argument('--feature-set', dest='feature_set', default='qm_rdkit', choices=FEATURE_SETS)
    args = parser.parse_args()
    params = DEFAULT_MPNN_PARAMS

    print(f"OOF-on-train ({K_FOLDS}-fold) for '{args.feature_set}' ...")
    oof = oof_train(args.feature_set, params)
    print("final train-on-all + 278-test ...")
    test = final_test(args.feature_set, params)

    out = pd.concat([oof, test], ignore_index=True)
    if not os.path.isdir(PATH_PRED_RES):
        os.makedirs(PATH_PRED_RES)
    outfile = PATH_PRED_RES + f"mpnn_oof_test_{args.feature_set}.parquet"
    out.to_parquet(outfile, index=False)
    print(f"\nwrote {out.shape} -> {outfile}")
    print("Upload to s3; join to partition_assignments_* by smiles for the per-partition family comparison.")


if __name__ == '__main__':
    main()
