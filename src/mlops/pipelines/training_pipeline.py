# from zenml.config import DockerSettings
# from zenml.integrations.constants import MLFLOW
# from zenml.pipelines import pipeline

from src.mlops.steps import load_data, clean_data

def train_pipeline():

    df_annual_sorted_after_2000, industry_mappings, df_company_info = load_data(
        data_path='/data/zhuanghao/MyGithub/MLOps-ForecastSys/data/input/union_ebitda_rev_cashflowfromoper_capex_merged_with_x_vars.csv',
        comp_path='/data/zhuanghao/MyGithub/MLOps-ForecastSys/data/input/Company Info.xlsx'
    )
    clean_data(df_annual_sorted_after_2000=df_annual_sorted_after_2000,
               industry_mappings=industry_mappings,
               df_company_info=df_company_info)


train_pipeline()