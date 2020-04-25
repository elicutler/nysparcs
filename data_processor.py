
import typing as T
import logging
import pandas as pd

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)


class DataProcessor(EnforceOverrides):
  
  def __init__(self, params) -> None:
    self.params = params
    
  @abstractmethod
  def process(self, inDF) -> pd.DataFrame:
    pass
  
  
class TorchDataProcessor(DataProcessor):
  
  def __init__(self, params) -> None:
    super().__init__(params)
    
  @overrides
  def process(self, inDF) -> pd.DataFrame:
    pass