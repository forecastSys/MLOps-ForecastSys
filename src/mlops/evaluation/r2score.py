from src.mlops.abstractions import EvaluationABC
from src.mlops.model import ModelLoader
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class R2Score(EvaluationABC):

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        r2 = r2_score(y_true, y_pred)

        return r2