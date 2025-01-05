from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os

class ModelABC(ABC):
    """
    Abstract base class for all models.
    """
    forecast_points = None

    @abstractmethod
    def train(self, X_train: pd.Dataframe, y_train: pd.Series,
                    id_bb_unique: str, x_reconstruction_type: str) -> None:
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
        pass

class UnivariateTSModelABC(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, training_points: pd.Series, forecast_points: int) -> list:
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass