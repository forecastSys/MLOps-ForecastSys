from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
# from zenml import step

from src.mlops.logger import LoggerDescriptor
from src.mlops.data_cleaning import DataCleaner, BBGDataMerger

def clean_data(df_annual_sorted_after_2000: pd.DataFrame,
               industry_mappings: list,
               df_company_info: pd.DataFrame) -> pd.DataFrame:

    df_merged, industry_cols_to_merge = DataCleaner(strategy=BBGDataMerger()).handle_data(
        df_annual_sorted_after_2000=df_annual_sorted_after_2000,
        industry_mappings=industry_mappings,
        df_company_info=df_company_info
    )