import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv() # 'envs/.env'
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    raise FileNotFoundError("Could not find a .env file in any parent directory")

ENV_DEVELOPMENT = os.environ["ENV_DEVELOPMENT"]

if ENV_DEVELOPMENT == "local":
    PATH_ROOT = "/Users/zhenguo/guo/code/tools/trans_learn/"
elif ENV_DEVELOPMENT == "lab":
    PATH_ROOT = "/home/jovyan/tools/trans_learn/"
else:
    PATH_ROOT = ""

PROJECT_BUCKET = "cdi-lab-workspaces"
PATH_TEMP_FILES = PATH_ROOT + "tests/"


@dataclass(frozen=True)
class DatasetSplitPaths:
    data: str
    fps: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    splits: Dict[str, DatasetSplitPaths]
    id_column_count: int
    target_column: Optional[str] = None
    drop_columns: Tuple[str, ...] = ()
    sep: str = "\t"
    remove_gas_solvent: bool = False
    solvent_length_limit: Optional[int] = None
    fill_identifier_columns: Tuple[str, ...] = ()
    split_decimal_identifier_columns: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = field(default_factory=tuple)


DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "abs": DatasetConfig(
        name="absorption",
        id_column_count=7,
        target_column="abs",
        splits={
            "train": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/abs_deep4chem_rdkit_fg_train.parquet",
                fps={
                    "defualt": "",
                },
            ),
            "test": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/abs_deep4chem_rdkit_fg_test.parquet",
                fps={
                    "defualt": "",
                },
            ),
        },
        notes=(
            "data without reaxys data"
            "got solvent smiles col but no solvent descriptors"
        ),
    ),
    "emi": DatasetConfig(
        name="emission",
        id_column_count=4,
        target_column="emission",
        splits={
            "train": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/emi_wo_reaxys_rdkit_fg_train.parquet",
                fps={
                    "defualt": "",
                },
            ),
            "test": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/emi_wo_reaxys_rdkit_fg_test.parquet",
                fps={
                    "defualt": "",
                },
            ),
        },
        notes=(
            "data without reaxys data"
            "got solvent smiles col but no solvent descriptors"
        ),
    ),
    "qy": DatasetConfig(
        name="quantum yield",
        id_column_count=6,
        target_column="qy",
        splits={
            "train": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/qy_wo_reaxys_rdkit_fg_train.parquet",
                fps={
                    "defualt": "",
                },
            ),
            "test": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/qy_wo_reaxys_rdkit_fg_test.parquet",
                fps={
                    "defualt": "",
                },
            ),
        },
        notes=(
            "data without reaxys data"
            "got solvent smiles col but no solvent descriptors"
        ),
    ),
    "extin": DatasetConfig(
        name="extinction coefficient",
        id_column_count=9,
        target_column="epsilon_log",
        splits={
            "train": DatasetSplitPaths(
                data="ts_project_1/data/extinction_abs/train_test/extin_rdkit_fg_train.parquet",
                fps={
                    "defualt": "",
                },
            ),
            "test": DatasetSplitPaths(
                data="ts_project_1/data/extinction_abs/train_test/extin_rdkit_fg_test.parquet",
                fps={
                    "defualt": "",
                },
            ),
        },
        notes=(
            "epsilon_log = np.log10(epsilon_canonical + 1). "
            "solvent affect extinction coefficient. "
            "lambda_abs can be molecular descriptor for the mode. to exclude it set id_column_count=10"
            "got solvent smiles col but no solvent descriptors"
        ),
    ),
    "biodeg": DatasetConfig(
        name="biodeg", # data without reaxys
        id_column_count=5,
        target_column="degradable",
        fill_identifier_columns=("reaxys_id", "cas", "inchi"),
        split_decimal_identifier_columns=("reaxys_id",),
        notes=(
            "size of df of fps and df of features are match.",
        ),
        splits={
            "train": DatasetSplitPaths(
                # with reaxys data: biodeg_fg_rdkit_train.csv, fps_mpnn_biodeg_train_9489.csv
                data="ts_project_1/data/biodegradation/training_data/biodeg_fg_rdkit_train_no_reaxys.parquet",
                fps={
                    "default": "ts_project_1/data/biodegradation/training_data/fps_mpnn_biodeg_train_no_reaxys.parquet",
                },
            ),
            "test": DatasetSplitPaths(
                # with reaxys data: biodeg_fg_rdkit_test.csv, fps_mpnn_biodeg_test_502.csv
                data="ts_project_1/data/biodegradation/training_data/biodeg_fg_rdkit_test_no_reaxys.parquet",
                fps={
                    "default": "ts_project_1/data/biodegradation/training_data/fps_mpnn_biodeg_test_no_reaxys.parquet",
                },
            ),
        },
    ),
    "biodeg_gwu": DatasetConfig(
        name="biodeg",
        id_column_count=10,
        target_column="degradable",
        notes=(
            "",
        ),
        splits={
            "train": DatasetSplitPaths(
                # with reaxys data: biodeg_fg_rdkit_train.csv, fps_mpnn_biodeg_train_9489.csv
                data="ts_project_1/data/biodegradation/GWU/train_test/biodeg_gwu_b2_train.parquet",
                fps={
                    "default": "",
                },
            ),
            "test": DatasetSplitPaths(
                # with reaxys data: biodeg_fg_rdkit_test.csv, fps_mpnn_biodeg_test_502.csv
                data="ts_project_1/data/biodegradation/GWU/train_test/biodeg_gwu_b2_test.parquet",
                fps={
                    "default": "",
                },
            ),
        },
    ),
    # following are backup data
    "qy_w_reaxys": DatasetConfig(
        name="qy",
        id_column_count=6,
        target_column="qy",
        splits={
            "train": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/deep4chem_qy_custom_fg_rdkit_sol_class_train.csv",
                fps={
                    "classification": "ts_project_1/data/fluorescence/training_data/fps_mpnn_qy_d4c_train_12249.csv",
                    "regression": "ts_project_1/data/fluorescence/training_data/fps_mpnn_qy_d4c_regression_12249.csv",
                },
            ),
            "test": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/deep4chem_qy_custom_fg_rdkit_sol_class_test.csv",
                fps={
                    "classification": "ts_project_1/data/fluorescence/training_data/fps_mpnn_qy_d4c_test_1361.csv",
                    "regression": "ts_project_1/data/fluorescence/training_data/fps_mpnn_qy_d4c_regression_1361.csv",
                },
            ),
        },
    ),
    "emi_w_reaxys": DatasetConfig(
        name="emi",
        id_column_count=4,
        target_column="emission",
        remove_gas_solvent=True,
        solvent_length_limit=50,
        splits={
            "train": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/emission_train_rdkit_descs.csv",
                fps={
                    "default": "ts_project_1/data/fluorescence/training_data/fps_mpnn_emi_train_23952.csv",
                },
            ),
            "test": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/emission_test_rdkit_descs.csv",
                fps={
                    "default": "ts_project_1/data/fluorescence/training_data/fps_mpnn_emi_test_2655.csv",
                },
            ),
        },
    ),
    "abs_w_reaxys": DatasetConfig(
        name="abs",
        id_column_count=4,
        target_column="abs",
        drop_columns=("emi", "lifetime", "qy"),
        remove_gas_solvent=True,
        solvent_length_limit=50,
        splits={
            "train": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/abs_train_rdkit_descs.csv",
                fps={
                    "default": "ts_project_1/data/fluorescence/training_data/fps_mpnn_abs_train_15267.csv",
                },
            ),
            "test": DatasetSplitPaths(
                data="ts_project_1/data/fluorescence/training_data/abs_test_rdkit_descs.csv",
                fps={
                    "default": "ts_project_1/data/fluorescence/training_data/fps_mpnn_abs_test_1701.csv",
                },
            ),
        },
    ),
    "extinction_valid": DatasetConfig(
        name="extinction",
        id_column_count=10,
        target_column="extinction",
        notes=(
            "to include col lambda_abs as a feature set id_column_count to 9",
        ),
        splits={
            "train": DatasetSplitPaths(
                # with reaxys data: biodeg_fg_rdkit_train.csv, fps_mpnn_biodeg_train_9489.csv
                data="ts_project_1/data/extinction_abs/train_test/extin_valid_rdkit_fg_train.parquet",
                fps={
                    "default": "",
                },
            ),
            "test": DatasetSplitPaths(
                # with reaxys data: biodeg_fg_rdkit_test.csv, fps_mpnn_biodeg_test_502.csv
                data="ts_project_1/data/extinction_abs/train_test/extin_valid_rdkit_fg_test.parquet",
                fps={
                    "default": "",
                },
            ),
        },
    ),
    "extinction_unique_smis": DatasetConfig(
        name="extinction",
        id_column_count=10,
        target_column="extinction",
        notes=(
            "to include col lambda_abs as a feature set id_column_count to 9",
        ),
        splits={
            "train": DatasetSplitPaths(
                # with reaxys data: biodeg_fg_rdkit_train.csv, fps_mpnn_biodeg_train_9489.csv
                data="ts_project_1/data/extinction_abs/train_test/extin_unique_smis_rdkit_fg_train.parquet",
                fps={
                    "default": "",
                },
            ),
            "test": DatasetSplitPaths(
                # with reaxys data: biodeg_fg_rdkit_test.csv, fps_mpnn_biodeg_test_502.csv
                data="ts_project_1/data/extinction_abs/train_test/extin_unique_smis_rdkit_fg_test.parquet",
                fps={
                    "default": "",
                },
            ),
        },
    ),
}
