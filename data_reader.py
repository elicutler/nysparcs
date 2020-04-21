
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
  
  def __init__(self, dataID):
    self.dataID = dataID
    
  @overrides
  def read(self):
    pass


class InternetDataReader(DataReader):
  
  def __init__(self, dataID):
    self.dataID = dataID


class DataReaderFactory:
  
  @staticmethod
  def make(dataLoc, dataID):
    
    if dataLoc == 'local':
      return LocalDataReader(dataID)
    
    raise ValueError(f'{dataLoc=} not recognized')