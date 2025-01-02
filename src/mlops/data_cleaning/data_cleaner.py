from src.mlops.data_cleaning.data_cleaner_strategy_abc import DataCleanerStrategyABC
from typing import Union
import pandas as pd

class DataCleaner:

    def __init__(self, strategy: DataCleanerStrategyABC):

        self.strategy = strategy

    def handle_data(self, **kwargs) -> Union[pd.DataFrame, pd.Series, list]:

        return self.strategy.handle_data(**kwargs)