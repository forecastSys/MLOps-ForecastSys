from src.mlops.logger import LoggerDescriptor
from src.mlops.configs import XYVariables
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd

class DataCleanerStrategyABC(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    logger = LoggerDescriptor()
    x_cols_to_process = XYVariables().X_SELECTED
    industry_cols = XYVariables().INDUSTRY

    @abstractmethod
    def handle_data(self, **kwargs) -> Union[pd.DataFrame, pd.Series, list]:
        pass

