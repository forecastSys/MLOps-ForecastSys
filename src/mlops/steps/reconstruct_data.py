from src.mlops.data_reconstruction import DataReconstructor, BBGDataReconstructionTS
from src.mlops.model import AR, ARIMA
from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated
import pandas as pd
import pickle
import os
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def reconstruct_data(companyID_df_dict: dict) -> Dict[str, Dict[str, Any]]:

    # companyID_df_postConstru_dict = DataReconstructor(strategy=BBGDataReconstructionTS()).handle_data(companyID_df_dict=companyID_df_dict,
    #                                                                                                   model=model)
    # Todo: remove all data reading and dumping - this is for development purpose
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/output'))
    file_path = os.path.join(base_dir, "companyID_df_postConstru_dict.pkl")
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing file
        with open(file_path, "rb") as file:
            companyID_df_postConstru_dict = pickle.load(file)
        print(f"File '{file_path}' loaded.")
    else:
        companyID_df_postConstru_ar_dict = DataReconstructor(strategy=BBGDataReconstructionTS()).handle_data(
            companyID_df_dict=companyID_df_dict,
            model=AR(),
            modify_dict_type='create')
        companyID_df_postConstru_dict = DataReconstructor(strategy=BBGDataReconstructionTS()).handle_data(
            companyID_df_dict=companyID_df_postConstru_ar_dict,
            model=ARIMA(),
            modify_dict_type='append')
        with open(file_path, "wb") as file:
            pickle.dump(companyID_df_postConstru_dict, file)
        print(f"File '{file_path}' created and saved.")
    return companyID_df_postConstru_dict