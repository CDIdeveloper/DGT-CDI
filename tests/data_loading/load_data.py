import json
from typing import Optional

import pandas as pd
import numpy as np

# from trans_learn.utils.molecule import sanitize_smis
from trans_learn.settings import DATASET_REGISTRY, PROJECT_BUCKET # PATH_ROOT, 
from trans_learn.utils.awstools import S3Handler

s3_handler = S3Handler()

# Utils
#region
def replace_nans_by_column(arr: np.ndarray) -> np.ndarray:
    """
    Given a 2D NumPy array, replace NaNs column-wise:
      - If an entire column is NaN, set all its entries to 0.
      - Otherwise, replace NaNs in that column with the column’s mean 
        computed over its non-NaN entries.
    arr : np.ndarray, shape (n_rows, n_cols)
        Input array (must be float‐typed to hold np.nan).
    Return
        np.ndarray, shape (n_rows, n_cols)
        A copy of `arr` where all NaNs have been replaced as specified.
    """
    # Work on a copy so we don’t modify the original
    result = arr.copy().astype(float)
    # Iterate over columns
    n_cols = result.shape[1]
    for j in range(n_cols):
        col = result[:, j]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            # no NaNs in this column → nothing to do
            continue
        if nan_mask.all():
            # entire column is NaN → replace all with 0
            result[:, j] = 0.0
        else:
            # partial NaNs → compute mean of non-NaN entries, fill NaNs with it
            mean_val = np.nanmean(col)
            col[nan_mask] = mean_val
            result[:, j] = col
    return result

def replace_nans_by_column_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a pandas DataFrame, replace NaNs column-wise:
      - If an entire column is NaN, set all its entries to 0.
      - Otherwise, replace NaNs in that column with the column’s mean 
        computed over its non-NaN entries.
    Returns a new DataFrame; the original is not modified.
    """
    result = df.copy()
    for col in result.columns:
        if result[col].isna().all():
            # entire column is NaN → fill with 0
            result[col] = 0
        else:
            # compute mean over non-NaN and fill
            mean_val = result[col].mean(skipna=True)
            result[col] = result[col].fillna(mean_val)
    return result

def remove_large_values(df):
    """remove inf and super large values"""
    mask_finite = np.isfinite(df).all(axis=1)
    # drop values too large for float64
    max64 = np.finfo(np.float64).max
    mask_not_too_big = (df.abs() <= max64).all(axis=1)
    # combine
    good = mask_finite & mask_not_too_big
    df = df[good].reset_index(drop=True)

    df = replace_nans_by_column_df(df)
    
    df = df.dropna()
    df.reset_index(inplace=True, drop=True)
    return df
#endregion

# dataloading function
#region
# DatasetLoader created after remove Reaxys data
class DatasetLoader:
    """
    Shared train/test data loader for property datasets.

    The current design centralizes dataset path/config metadata in
    `trans_learn.settings.DATASET_REGISTRY` and keeps train/test loading
    behavior consistent across datasets used by ChemProp and smaller ML models.

    NOTE currently "biodeg" means data without reaxys data, see DATASET_REGISTRY
    """

    def __init__(self, dataset_name: str, storage_handler: Optional[S3Handler] = None):
        self.dataset_name = dataset_name
        self.storage_handler = storage_handler or s3_handler
        self.config = self._get_dataset_config(dataset_name)

    def load_train(
        self,
        cols_selected=None,
        col_select_labels=None,
        normlize=False,
        load_fps=False,
        fps_variant="default",
    ):
        return self.load_split(
            split="train",
            cols_selected=cols_selected,
            col_select_labels=col_select_labels,
            normlize=normlize,
            load_fps=load_fps,
            fps_variant=fps_variant,
        )

    def load_test(
        self,
        cols_selected=None,
        col_select_labels=None,
        normlize=False,
        load_fps=False,
        fps_variant="default",
    ):
        return self.load_split(
            split="test",
            cols_selected=cols_selected,
            col_select_labels=col_select_labels,
            normlize=normlize,
            load_fps=load_fps,
            fps_variant=fps_variant,
        )

    def load_split(
        self,
        split,
        cols_selected=None,
        col_select_labels=None,
        normlize=False,
        load_fps=False,
        fps_variant="default",
    ):
        self._validate_split(split)
        df = self._read_split_data(split)
        df = self._preprocess_dataframe(df)
        df_ids_ys, df_descs = self._split_ids_and_descs(df)

        selected_columns = self._resolve_selected_columns(
            df_descs.columns,
            cols_selected=cols_selected,
            col_select_labels=col_select_labels,
        )
        if selected_columns is not None:
            df_descs = df_descs.loc[:, selected_columns]

        if normlize:
            df_descs = (df_descs - df_descs.min()) / (df_descs.max() - df_descs.min())
            if self.dataset_name in {
                "emi", "abs", "biodeg", "biodeg_gwu", "extinction_valid",
            }:
                df_descs = remove_large_values(df_descs)

        df_descs = replace_nans_by_column_df(df_descs)

        if load_fps:
            data_fps = self._load_fps(split=split, fps_variant=fps_variant)
            if data_fps is not None:
                if normlize:
                    data_fps = (data_fps - data_fps.min()) / (data_fps.max() - data_fps.min())
                    if self.dataset_name in {"emi", "abs", "biodeg", "biodeg_gwu"}:
                        data_fps = remove_large_values(data_fps)
                df_descs = pd.concat([df_descs, data_fps], axis=1)
        
        return {
            "ids_ys": df_ids_ys,
            "descs": df_descs,
        }

    def _get_dataset_config(self, dataset_name):
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return DATASET_REGISTRY[dataset_name]

    def _validate_split(self, split):
        if split not in self.config.splits:
            raise ValueError(f"Unsupported split '{split}' for dataset '{self.dataset_name}'")

    def _read_split_data(self, split):
        split_config = self.config.splits[split]
        infile_name = split_config.data
        if infile_name.endswith(".parquet"):
            return pd.read_parquet(f"s3://{PROJECT_BUCKET}/{infile_name}")
        with self.storage_handler.open_file(infile_name) as infile:
            return pd.read_csv(infile, sep=self.config.sep)

    def _preprocess_dataframe(self, df):
        df = df.copy()

        # following actions depends on dataset_name and correspoinding config
        # just check if config.<item> is in config under the dataset_name
        # only for dataset_name: biodeg
        for col in self.config.fill_identifier_columns:
            if col in df.columns:
                df[col] = df[col].fillna("")

        for col in self.config.split_decimal_identifier_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: x.split(".")[0])

        if self.config.remove_gas_solvent and "solvent" in df.columns:
            df = df[df["solvent"] != "gas"]

        if self.config.solvent_length_limit is not None and "solvent" in df.columns:
            df = df[df["solvent"].astype(str).str.len() < self.config.solvent_length_limit]

        if self.config.drop_columns:
            cols_to_drop = [col for col in self.config.drop_columns if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

        if self.config.target_column and self.config.target_column in df.columns:
            df = df[df[self.config.target_column].notna()]

        if self.dataset_name in {"biodeg",}:
            df = df.dropna()

        df.reset_index(inplace=True, drop=True)
        return df

    def _split_ids_and_descs(self, df):
        df_ids_ys = df.iloc[:, : self.config.id_column_count]
        df_descs = df.iloc[:, self.config.id_column_count :]
        return df_ids_ys, df_descs

    def _resolve_selected_columns(self, columns, cols_selected=None, col_select_labels=None):
        if cols_selected is None and col_select_labels is None:
            return None

        selected_columns = []
        if cols_selected is not None:
            selected_columns.extend(cols_selected)

        if col_select_labels is not None:
            for col_select_label in col_select_labels:
                selected_columns.extend(
                    [col for col in columns if col_select_label in col and col not in selected_columns]
                )

        return selected_columns

    def _load_fps(self, split, fps_variant="default"):
        split_config = self.config.splits[split]
        if not split_config.fps:
            return None

        fps_path = split_config.fps.get(fps_variant)
        if fps_path is None and "default" in split_config.fps:
            fps_path = split_config.fps["default"]

        if not fps_path:
            # Implementation needed: decide whether to hard-fail or silently skip
            return None

        if self.dataset_name in {"biodeg", "biodeg_gwu"}:
            # Implementation needed: current legacy code notes row mismatch in biodeg FPS files
            return None

        with self.storage_handler.open_file(fps_path) as infile:
            return pd.read_csv(infile, sep=self.config.sep)

def load_split_data(split, dataset_name, col_select_labels=None):
    """
    Args:
        split: str, 'train' or 'test'
        dataset_name: str, name of dataset: "emi", "abs", "biodeg", "biodeg_gwu"
        col_select_labels: list of str, select col by labels, None to select all
            iterate the strings and check and select if column name contains the string
            example: '_gwu', 'rdkit', '_fg'
    return:
        dict of pd.DataFrame
    """
    data_loader = DatasetLoader(dataset_name)
    if split == "train":
        data_to_load = data_loader.load_train(
            col_select_labels=col_select_labels, load_fps=False
        )
    elif split == "test":
        data_to_load = data_loader.load_test(
            col_select_labels=col_select_labels, load_fps=False
        )
    else:
        raise ValueError(f"Unsupported split: {split}")
    return data_to_load

class GetSelectedFeatures:
    """
    Get selected features from a list of features
    Args:
        features: list, list of features
        selected_features: list, list of selected features
    """

    def qy_features(self, coef_thresh=0.05):
        """
        Args:
            coef_thresh: float, 
                threshold for selecting features = coef_thresh*max(abs(SHAP_values))
                can put 0.0 to select features with non-zero SHAP values
        Return:
            dataframe of features and feature_indices
        """
        infile_name = 'ts_project_1/data/fluorescence/features/feature_rank_shap_qy_d4c.csv'
        with s3_handler.open_file(infile_name) as infile:
            df_features = pd.read_csv(infile, sep='\t')
            # shape: (361, 5)
            # 'features', 'mean_shap_abs', 'mean_shap', 'direction', 'feature_indices'
        # max should be the first one in the df
        # min is 0.0
        thresh = df_features['mean_shap_abs'].max() * coef_thresh 
        if thresh !=0:
            df_features_selected = df_features[
                df_features['mean_shap_abs'] >= thresh][['features', 'feature_indices']
            ]
        else:
            df_features_selected = df_features[
                df_features['mean_shap_abs'] > thresh][['features', 'feature_indices']
            ]
        # shape: (179, 2) when coef_thresh=0.05, (323, 2) when coef_thresh=0.0
        return df_features_selected

    def biodeg_features_wo_reaxys(self, split, col_select_labels=['rdkit', '_fg']):
        """data without reaxys data
        Args:
            split: str, 'train', 'test'
            coef_thresh: float, 
                threshold for selecting features = coef_thresh*max(abs(SHAP_values))
                can put 0.0 to select features with non-zero SHAP values
        Return:
            list of column names
        """
        data_biodeg = load_split_data(
            split, dataset_name="biodeg", col_select_labels=col_select_labels
        )
        return data_biodeg['descs'].columns.to_list()

    def biodeg_features_wo_reaxys_coef(self, coef_thresh=0.05, feature_set='rdkit_fg'):
        if feature_set == 'rdkit_fg':
            infile_name = 'ts_project_1/data/biodegradation/features/feature_rank_shap_no_reaxys_rdkit.csv'
        with s3_handler.open_file(infile_name) as infile:
            df_features = pd.read_csv(infile, sep='\t')
        thresh = df_features['mean_shap_abs'].max() * coef_thresh 
        if thresh !=0:
            df_features_selected = df_features[
                df_features['mean_shap_abs'] >= thresh][['features', 'feature_indices']
            ]
        else:
            df_features_selected = df_features[
                df_features['mean_shap_abs'] > thresh][['features', 'feature_indices']
            ]
        return df_features_selected
    
    def biodeg_features(
        self, split, col_select_labels=['rdkit', '_fg'], dataset_name="biodeg_gwu"
    ):
        """
        Args:
            split: str, 'train', 'test'
            col_select_labels: list, ['_gwu', 'rdkit', '_fg'], get cols by labels in colnames
            data_set: str, 'biodeg_gwu': gwu qm batch 2 and rdkit, 'biodeg' (both without reaxys)
        Return:
            list of column names
        """
        data_biodeg = load_split_data(
            split, dataset_name=dataset_name, col_select_labels=col_select_labels
        )
        return data_biodeg['descs'].columns.to_list()

    def biodeg_gwu_features(self, coef_thresh=0.05, feature_set='qm'):
        """
        Args:
            coef_thresh: float, 
                threshold for selecting features = coef_thresh*max(abs(SHAP_values))
                can put 0.0 to select features with non-zero SHAP values
            feature_set: str, 'qm', 'qm_rdkit', determines feature analysis result, qm only or qm and rdkit
        Return:
            dataframe of features and feature_indices
        """
        # b1 batch 1
        # infile_name = 'ts_project_1/data/biodegradation/features/feature_rank_shap_biodeg_QM_b1.csv'
        if feature_set == 'qm':
            infile_name = 'ts_project_1/data/biodegradation/features/GWU/feature_rank_shap_gwu_b2_qm.csv'
        elif feature_set == 'rdkit':
            infile_name = 'ts_project_1/data/biodegradation/features/GWU/feature_rank_shap_gwu_b2_rdkit.csv'
        elif feature_set == 'qm_rdkit':
            infile_name = 'ts_project_1/data/biodegradation/features/GWU/feature_rank_shap_gwu_b2_qm_rdkit.csv'
        with s3_handler.open_file(infile_name) as infile:
            df_features = pd.read_csv(infile, sep='\t')
            # shape: (99, 5)
            # 'features', 'mean_shap_abs', 'mean_shap', 'direction', 'feature_indices'
        # max should be the first one in the df
        # min is 0.0
        thresh = df_features['mean_shap_abs'].max() * coef_thresh 
        if thresh !=0:
            df_features_selected = df_features[
                df_features['mean_shap_abs'] >= thresh][['features', 'feature_indices']
            ]
        else:
            df_features_selected = df_features[
                df_features['mean_shap_abs'] > thresh][['features', 'feature_indices']
            ]
        # shape: (37, 2) when coef_thresh=0.05, (71, 2) when coef_thresh=0.0
        return df_features_selected



#endregion

