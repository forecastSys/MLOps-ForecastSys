from src.mlops.configs import Variables
from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated
import pandas as pd
import pickle
import os
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def load_data(id_bb_unique: str, y: str, year: int) -> Dict[str, Any]:
    # Todo: remove all data reading and dumping - this is for development purpose
    file_path = "../../../data/output/companyID_df_postConstru_dict.pkl"
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing file
        with open(file_path, "rb") as file:
            companyID_df_postConstru_dict = pickle.load(file)
        print(f"File '{file_path}' loaded.")
    else:
        raise FileNotFoundError(f"File '{file_path}' not found.")

    df_X_test = companyID_df_postConstru_dict[id_bb_unique]['df_test_predby_ar_w_category']
    print("-"*100)
    print(year)
    print("-"*100)
    df_X_test = df_X_test[df_X_test.Year.apply(lambda x: int(x.year) == year)][Variables.x_cols_to_process]
    X = df_X_test.drop(columns=[y])
    return X