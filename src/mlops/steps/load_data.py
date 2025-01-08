from src.mlops.data_loading import DataLoader
from src.mlops.logger import LoggerDescriptor
from typing import Tuple
import pandas as pd
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name)
def load_data(data_path, comp_path) -> Tuple[pd.DataFrame, list, pd.DataFrame]:

    df_annual_sorted_after_2000, industry_mappings, df_company_info = DataLoader().load_data(
        data_path=data_path, comp_path=comp_path
    )
    return df_annual_sorted_after_2000, industry_mappings, df_company_info

@step(experiment_tracker=experiment_tracker.name)
def load_intermediate_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Todo: remove all data reading and dumping - this is for development purpose
    df = pd.read_csv('../../../data/output/union_processed_imputed_80_20.csv')
    df_ground_truth = pd.read_csv('../../../data/output/union_processed_groundtruth_80_20.csv')
    return df, df_ground_truth
