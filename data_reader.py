
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)


class DataReader(EnforceOverrides):
  
  @abstractmethod
  def read(self):
    pass


class LocalDataReader(DataReader):
  
  def __init__(self, params) -> None:
    self.params = params 
    
  @overrides
  def read(self):
    pass


class DataReaderFactory:
  
  @staticmethod
  def make(params) -> T.Type[DataReader]:
    
    dataLoc = 'local' if params['local_data_path'] is not None else 'internet'
    
    if dataLoc == 'local':
      return LocalDataReader(params.copy())
    
    raise ValueError(f'{dataLoc=} not recognized')