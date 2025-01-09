# from scripts.pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
from src.mlops.steps_deployment import load_data, load_registered_model
import click
from typing import cast
import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService


def run_deployment(model_type, id_bb_unique, y, year):
    data = load_data()
    model_name = model_type + id_bb_unique + y
    model = load_registered_model(model_name=model_name).get_model()
    df_X_test = data[id_bb_unique]['df_test_predby_arima']
    df_X_test = df_X_test.drop(columns=[y])
    df_X_test
    y_pred = model.predict(df_X_test)

    return y_pred

