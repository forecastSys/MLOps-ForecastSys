from src.mlops.abstractions import EvaluationABC
from src.mlops.model import ModelLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

class RMSE(EvaluationABC):

    def evaluate(self,
                 df_X_test: pd.DataFrame,
                 df_y_true: pd.Series,
                 mlflow_model_name: str,
                 mlflow_model_run_id: str) -> float:

        model, model_uri = ModelLoader.load_model(mlflow_model_name, mlflow_model_run_id)
        y_pred = model.predict(df_X_test)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        with mlflow.start_run(run_id=mlflow_model_run_id):
            mlflow.log_metric("rmse", rmse)

        mlflow.register_model(model_uri=model_uri, name=mlflow_model_name)
        print(f"Model registered as {registered_model_name}")
        return rmse