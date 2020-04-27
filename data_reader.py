
import typing as T
import logging
import pandas as pd
import json

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from sodapy import Socrata

logger = logging.getLogger(__name__)


class DataReader(EnforceOverrides):
  
  def __init__(self, params) -> None:
    self.params = params.copy()
  
  @abstractmethod
  def read(self) -> pd.DataFrame:
    pass
  
  @abstractmethod
  def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
    pass
  

class LocalDataReader(DataReader):
  
  def __init__(self, params) -> None:
    super().__init__(params)
    
  @overrides
  def read(self) -> pd.DataFrame:
    logger.info('Reading data from local path...')
    
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

  @overrides
  def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
    colNames = pd.read_csv(self.params['local_data_path'], nrows=0).columns
    df = pd.read_csv(
      self.params['local_data_path'], skiprows=startRow-1, nrows=numRows,
      header=None, names=colNames
    )
    return df
  

class SocrataDataReader(DataReader):
  
  def __init__(self, params) -> None:
    super().__init__(params)
    self.socrataConn = self._establishSocrataConn()
    
  @overrides
  def read(self) -> pd.DataFrame:
    logger.info('Reading data from socrata...')
    
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
    
  @overrides
  def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
    socrataDataKey = self.params['socrata_data_key']
    dataRecs = self.socrataConn.get(
      socrataDataKey, order=':id', offset=startRow, limit=numRows
    )
    df = pd.DataFrame.from_records(dataRecs)
    return df
  
  def _establishSocrataConn(self) -> Socrata:
    with open('config/secrets.json', 'r') as f:
      appToken = json.load(f)['socrata']['app_token']
    return Socrata('health.data.ny.gov', appToken)
                

class DataReaderFactory:
  
  @staticmethod
  def make(params) -> T.Type[DataReader]:
    
    dataLoc = 'local' if params['local_data_path'] is not None else 'socrata'
    
    if dataLoc == 'local':
      return LocalDataReader(params)
    elif dataLoc == 'socrata':
      return SocrataDataReader(params)
    
    raise ValueError(f'{dataLoc=} not recognized')