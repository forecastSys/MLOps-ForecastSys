from src.mlops.abstractions import ModelABC
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import mlflow
import mlflow.sklearn

class RFRegression(ModelABC):

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              id_bb_unique: str, x_reconstruction_type: str) -> None:

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            max_features=None,
            random_state=42
        )

        # Define the parameter grid to search over
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'max_features': ['sqrt', 'log2', None],
        }

        cv = min(5, len(X_train))

        with mlflow.start_run(run_name=f"RandomForest_{id_bb_unique}_{x_reconstruction_type}") as run:
            if cv >= 2:
                grid_search = RandomizedSearchCV(
                    estimator=rf,
                    param_distributions=param_grid,
                    n_iter=20,
                    cv=cv,
                    n_jobs=-1,
                    verbose=False,
                    random_state=42,
                    scoring='neg_root_mean_squared_error',
                )
                # Fit the model using grid search on training data
                grid_search.fit(X_train, y_train)

                # Get the best parameters and log them
                best_params = grid_search.best_params_
                mlflow.log_params(best_params)

                # Use the best estimator from grid search
                best_rf = grid_search.best_estimator_

                # Log metrics from cross-validation
                best_score = grid_search.best_score_
                mlflow.log_metric("best_neg_rmse", best_score)

            else:
                # Not enough data for cross-validation; fit the model directly
                rf.fit(X_train, y_train)
                best_rf = rf  # Assign the directly fitted model as the best model

                # Log default parameters
                default_params = {
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "max_features": None
                }
                mlflow.log_params(default_params)

            # Log the model to MLflow
            model_name = f"{RFRegression.__name__}_{id_bb_unique}_{x_reconstruction_type}"
            mlflow.sklearn.log_model(best_rf, model_name=model_name)

            model_uri = f"runs:/{run.info.run_id}/{model_name}"
            mlflow.register_model(model_uri=model_uri, name=model_name)

            print(f"Model for company: {id_bb_unique}, "
                  f"x_reconstruction_type: {x_reconstruction_type} "
                  f"registered with model name **{model_name}**")

            return best_rf