import mlflow
from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated
import pandas as pd
import pickle
import os
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def load_registered_model(model_name, id_bb_unique, y, model_version="latest") -> Union[Any]:

    model_registered_name = model_name + '_' + id_bb_unique + '_' + y
    model_registered_name = 'LGBRegression_EQ0000000000182032_EBITDA'
    model_uri = f"mlruns/models:/{model_registered_name}/{model_version}"
    if "RF" in model_name:
        return mlflow.sklearn.load_model(model_uri)
    elif "LGB" in model_name:
        return mlflow.lightgbm.load_model(model_uri)
