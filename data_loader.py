
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoaderFactory:
  
  @staticmethod
  def make(dataLoc, dataID) -> T.Type[DataLoader]:
    
    if dataLoc == 'local':
      return LocalDataLoader(dataD)
    
    raise ValueError(f'{dataLoc=} not recognized')
    
    
class DataLoader(EnforceOverrides):
  
  @abstractmethod
  def load(self):
    pass