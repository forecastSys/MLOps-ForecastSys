from src.mlops.abstractions import ModelABC, UnivariateTSModelABC
from abc import ABC, abstractmethod
from typing import Union, Tuple, List
import numpy as np
import pandas as pd

class ModelCaller(ABC):

    def __init__(self, strategy: Union[ModelABC, UnivariateTSModelABC]):
        self.strategy = strategy

    def train(self, **kwargs) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        return self.strategy.train(**kwargs)