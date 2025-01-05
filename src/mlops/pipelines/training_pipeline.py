from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
import pandas as pd
import os

from src.mlops.steps import (
    load_data,
    load_intermediate_data,
    clean_data,
    split_data,
    impute_data,
    partition_data,
    reconstruct_data
)

# docker_settings = DockerSettings(required_integrations=[MLFLOW])
# @pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline():

    ### --------------------------- Data Preprocessing ------------------------ ###
    # df_annual_sorted_after_2000, industry_mappings, df_company_info = load_data(
    #     data_path='/data/zhuanghao/MyGithub/MLOps-ForecastSys/data/input/union_ebitda_rev_cashflowfromoper_capex_merged_with_x_vars.csv',
    #     comp_path='/data/zhuanghao/MyGithub/MLOps-ForecastSys/data/input/Company Info.xlsx'
    # )
    #
    # df_combined = clean_data(df_annual_sorted_after_2000=df_annual_sorted_after_2000,
    #            industry_mappings=industry_mappings,
    #            df_company_info=df_company_info)
    # train_df_list, ground_truth_list = split_data(df_combined=df_combined)
    # df, df_ground_truth = impute_data(train_df_list=train_df_list, ground_truth_list=ground_truth_list)

    # Todo: remove all data reading and dumping - this is for development purpose
    df, df_ground_truth = load_intermediate_data()
    companyID_df_dict = partition_data(df, df_ground_truth)
    companyID_df_postConstru_dict = reconstruct_data(companyID_df_dict)

    # Todo: ### --------------------------- Run univariate ts pipeline ------------------------ ###

    ### --------------------------- Run multi-variate ts pipeline ------------------------ ###


    print(companyID_df_postConstru_dict)
train_pipeline()