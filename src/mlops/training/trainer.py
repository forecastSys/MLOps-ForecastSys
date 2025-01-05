from src.mlops.abstractions import TrainingABC
from typing import Union
import pandas as pd

class Trainer(TrainingABC):

    def __init__(self, strategy: DataPreprocessingStrategyABC):
        self.strategy = strategy

    def run_training(self, **kwargs) -> Union[pd.DataFrame, pd.Series, list]:
        return self.strategy.handle_data(**kwargs)