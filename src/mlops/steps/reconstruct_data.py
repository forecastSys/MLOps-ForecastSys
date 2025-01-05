from src.mlops.data_reconstruction import DataReconstructor, BBGDataReconstructionTS
from src.mlops.model import AR, ARIMA
from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated
import pandas as pd
import pickle
import os
from zenml import step

# @step
def reconstruct_data(companyID_df_dict: dict) -> Dict[str, Dict[str, Any]]:

    # companyID_df_postConstru_dict = DataReconstructor(strategy=BBGDataReconstructionTS()).handle_data(companyID_df_dict=companyID_df_dict,
    #                                                                                                   model=model)
    # Todo: remove all data reading and dumping - this is for development purpose
    file_path = "../../../data/output/companyID_df_postConstru_dict.pkl"
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing file
        with open(file_path, "rb") as file:
            companyID_df_postConstru_dict = pickle.load(file)
        print(f"File '{file_path}' loaded.")
    else:
        companyID_df_postConstru_ar_dict = DataReconstructor(strategy=BBGDataReconstructionTS()).handle_data(companyID_df_dict=companyID_df_dict,
                                                                                                      model=AR())
        companyID_df_postConstru_dict = DataReconstructor(strategy=BBGDataReconstructionTS()).handle_data(companyID_df_dict=companyID_df_postConstru_ar_dict,
                                                                                                      model=ARIMA())
        with open("../../../data/output/companyID_df_postConstru_dict.pkl", "wb") as file:
            pickle.dump(companyID_df_postConstru_dict, file)
        print(f"File '{file_path}' created and saved.")
    return companyID_df_postConstru_dict