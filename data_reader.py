
import typing as T
import logging
import pandas as pd

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)


class DataReader(EnforceOverrides):
  
  def __init__(self, params) -> None:
    self.params = params 
  
  @abstractmethod
  def readTrainRange(self) -> pd.DataFrame:
    pass
  
  @abstractmethod
  def readTestRange(self) -> pd.DataFrame:
    pass
  


class LocalDataReader(DataReader):
  
  def __init__(self, params) -> None:
    super().__init__(params)
    
  @overrides
  def readTrainRange(self) -> pd.DataFrame:
    startRow = self.params['train_range'][0]
    numRows = self.params['train_range'][1] - self.params['train_range'][0]
    df = self._readNumRowsFromStartRow(startRow, numRows)
    return df

  @overrides
  def readTestRange(self) -> pd.DataFrame:
    pass
    
  def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
    colNames = pd.read_csv(self.params['local_data_path'], nrows=0).columns
    df = pd.read_csv(
      self.params['local_data_path'], skiprows=startRow-1, nrows=numRows,
      header=None, names=colNames
    )
    return df
    


class DataReaderFactory:
  
  @staticmethod
  def make(params) -> T.Type[DataReader]:
    
    dataLoc = 'local' if params['local_data_path'] is not None else 'internet'
    
    if dataLoc == 'local':
      return LocalDataReader(params.copy())
    
    raise ValueError(f'{dataLoc=} not recognized')