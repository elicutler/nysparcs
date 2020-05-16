
import typing as T
import logging
import pandas as pd
import json

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from sodapy import Socrata

logger = logging.getLogger(__name__)


class DataReader(EnforceOverrides):
  
  def __init__(self, params):
    self.params = params.copy()
  
  @abstractmethod
  def read(self) -> pd.DataFrame:
    pass
  
  @abstractmethod
  def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
    pass
  

class LocalDataReader(DataReader):
  
  def __init__(self, params):
    super().__init__(params)
    
  @overrides
  def read(self) -> pd.DataFrame:
    logger.info('Reading data from local path...')
    
    trainStart, trainEnd = self.params['train_range']
    trainDF = self._readNumRowsFromStartRow(
      trainStart, trainEnd-trainStart
    )
    trainDF['train_val'] = 'train'
    
    valStart, valEnd = self.params['val_range']
    valDF = self._readNumRowsFromStartRow(
      valStart, valEnd-valStart
    )
    valDF['train_val'] = 'val'
    
    df = trainDF.append(valDF)
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
  
  def __init__(self, params):
    super().__init__(params)
    self.socrataConn = self._establishSocrataConn()
    
  @overrides
  def read(self) -> pd.DataFrame:
    logger.info('Reading data from socrata...')
    
    trainStart, trainEnd = self.params['train_range']
    trainDF = self._readNumRowsFromStartRow(
      trainStart, trainEnd-trainStart
    )
    trainDF['train_val'] = 'train'
    
    valStart, valEnd = self.params['val_range']
    valDF = self._readNumRowsFromStartRow(
      valStart, valEnd-valStart
    )
    valDF['train_val'] = 'val'
    
    df = trainDF.append(valDF)
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
  
  
class S3DataReader(DataReader):
  
  def __init__(self, params):
    super().__init__(params)
    
#   @overrides
#   def read(self) -> pd.DataFrame:
#     logger.info('Reading data from local path...')
    
#     trainStart, trainEnd = self.params['train_range']
#     trainDF = self._readNumRowsFromStartRow(
#       trainStart, trainEnd-trainStart
#     )
#     trainDF['train_val'] = 'train'
    
#     valStart, valEnd = self.params['val_range']
#     valDF = self._readNumRowsFromStartRow(
#       valStart, valEnd-valStart
#     )
#     valDF['train_val'] = 'val'
    
#     df = trainDF.append(valDF)
#     return df

#   @overrides
#   def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
#     colNames = pd.read_csv(self.params['local_data_path'], nrows=0).columns
#     df = pd.read_csv(
#       self.params['local_data_path'], skiprows=startRow-1, nrows=numRows,
#       header=None, names=colNames
#     )
#     return df
                

class DataReaderFactory:
  
  @staticmethod
  def make(params) -> T.Type[DataReader]:
    
    dataLoc = 'local' if params['local_data_path'] is not None else 'socrata'
    
    if dataLoc == 'local':
      dataReader = LocalDataReader
      
    elif dataLoc == 'socrata':
      dataReader = SocrataDataReader
      
    elif dataLoc == 's3':
      dataReader = S3DataReaderi
      
    else:
      raise ValueError(f'{dataLoc=} not recognized')
      
    return dataReader(params)