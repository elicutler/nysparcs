
import typing as T
import logging
import pandas as pd

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)


class DataReader(EnforceOverrides):
  
  def __init__(self, params) -> None:
    self.params = params.copy()
  
  @abstractmethod
  def read(self) -> pd.DataFrame:
    pass
  

class LocalDataReader(DataReader):
  
  def __init__(self, params) -> None:
    super().__init__(params)
    
  @overrides
  def read(self) -> pd.DataFrame:
    logger.info('Reading data...')
    
    trainStart, trainEnd = self.params['train_range']
    trainDF = self._readNumRowsFromStartRow(
      trainStart, trainEnd-trainStart
    )
    trainDF['train_test'] = 'train'
    
    testStart, testEnd = self.params['test_range']
    testDF = self._readNumRowsFromStartRow(
      testStart, testEnd-testStart
    )
    testDF['train_test'] = 'test'
    
    df = trainDF.append(testDF)
    return df

  def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
    colNames = pd.read_csv(self.params['local_data_path'], nrows=0).columns
    df = pd.read_csv(
      self.params['local_data_path'], skiprows=startRow-1, nrows=numRows,
      header=None, names=colNames
    )
    return df
  

class CloudDataReader(DataReader):
  
  def __init__(self, params) -> None:
    super().__init__(params)
    
  @overrides
  def read(self) -> pd.DataFrame:
    pass
    

class DataReaderFactory:
  
  @staticmethod
  def make(params) -> T.Type[DataReader]:
    
    dataLoc = 'local' if params['local_data_path'] is not None else 'cloud'
    
    if dataLoc == 'local':
      return LocalDataReader(params)
    elif dataLoc == 'cloud':
      return CloudDataReader(params)
    
    raise ValueError(f'{dataLoc=} not recognized')