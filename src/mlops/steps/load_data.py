import pandas as pd
# from zenml import step
from src.mlops.data_loading import DataLoader
from src.mlops.logger import LoggerDescriptor

def load_data(data_path, comp_path) -> pd.DataFrame:

    df_annual_sorted_after_2000, industry_mappings, df_company_info = DataLoader().load_data(
        data_path=data_path, comp_path=comp_path
    )
    return df_annual_sorted_after_2000, industry_mappings, df_company_info