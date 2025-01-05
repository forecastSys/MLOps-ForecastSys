from src.mlops.data_partitioning import DataPartitioner, BBGPostImputeDataPrep, BBGDataPartitioningCompany
from typing import Tuple, Dict, Any
from typing_extensions import Annotated
import pandas as pd
from zenml import step

# @step
def partition_data(df: pd.DataFrame, df_ground_truth: pd.DataFrame) \
        -> Dict[str, Dict[str, Any]]:

    df, df_x, df_ground_truth, df_industry_train, df_test, nan_rows = \
        DataPartitioner(strategy=BBGPostImputeDataPrep()).handle_data(df=df, df_ground_truth=df_ground_truth)

    companyID_df_dict = DataPartitioner(strategy=BBGDataPartitioningCompany()).handle_data(df=df, df_ground_truth=df_ground_truth)

    return companyID_df_dict