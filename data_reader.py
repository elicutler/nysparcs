
import typing as T
import logging
import pandas as pd
import json
import os

from pathlib import Path
from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from sodapy import Socrata
from safe_dict import SafeDict

logger = logging.getLogger(__name__)


class DataReader(EnforceOverrides):
  
  def __init__(self, trainParams: SafeDict):
    self.trainParams = trainParams.copy()
  
  @abstractmethod
  def read(self) -> pd.DataFrame:
    pass
  
  @abstractmethod
  def _readNumRowsFromStartRow(
    self, startRow: int, numRows: int
  ) -> pd.DataFrame:
    pass
  

class LocalOrS3DataReader(DataReader):
  
  def __init__(self, trainParams: SafeDict):
    super().__init__(trainParams)
    
    self.dataLoc = (
      'local' if self.trainParams['local_data_path'] is not None else 's3'
    )
    
    if self.dataLoc == 'local':
      self.dataPath = Path(trainParams['local_data_path'])
    
    elif self.dataLoc == 's3':
      s3Bucket = os.getenv('S3_BUCKET')
      s3Prefix = Path(f's3://{s3Bucket}/nysparcs/')
      self.dataPath = s3Prefix/trainParams['s3_data_path']
    
  @overrides
  def read(self) -> pd.DataFrame:
    logger.info(f'Reading data from {self.dataLoc} path...')
    
    trainStart, trainEnd = self.trainParams['train_range']
    trainDF = self._readNumRowsFromStartRow(
      trainStart, trainEnd-trainStart
    )
    trainDF['train_val'] = 'train'
    
    valStart, valEnd = self.trainParams['val_range']
    valDF = self._readNumRowsFromStartRow(
      valStart, valEnd-valStart
    )
    valDF['train_val'] = 'val'
    
    df = trainDF.append(valDF)
    return df

  @overrides
  def _readNumRowsFromStartRow(
    self, startRow: int, numRows: int
  ) -> pd.DataFrame:
    
    colNames = pd.read_csv(self.dataPath, nrows=0).columns
    df = pd.read_csv(
      self.dataPath, skiprows=startRow-1, nrows=numRows,
      header=None, names=colNames
    )
    return df
  

class SocrataDataReader(DataReader):
  
  def __init__(self, trainParams: SafeDict):
    super().__init__(trainParams)
    self.socrataConn = self._establishSocrataConn()
    
  @overrides
  def read(self) -> pd.DataFrame:
    logger.info('Reading data from socrata...')
    
    trainStart, trainEnd = self.trainParams['train_range']
    trainDF = self._readNumRowsFromStartRow(
      trainStart, trainEnd-trainStart
    )
    trainDF['train_val'] = 'train'
    
    valStart, valEnd = self.trainParams['val_range']
    valDF = self._readNumRowsFromStartRow(
      valStart, valEnd-valStart
    )
    valDF['train_val'] = 'val'
    
    df = trainDF.append(valDF)
    return df
    
  @overrides
  def _readNumRowsFromStartRow(
    self, startRow: int, numRows: int
  ) -> pd.DataFrame:
    
    socrataDataKey = self.trainParams['socrata_data_key']
    dataRecs = self.socrataConn.get(
      socrataDataKey, order=':id', offset=startRow, limit=numRows
    )
    df = pd.DataFrame.from_records(dataRecs)
    return df
  
  def _establishSocrataConn(self) -> Socrata:
    appToken = os.getenv('SOCRATA_APP_TOKEN')
    return Socrata('health.data.ny.gov', appToken)
                

class DataReaderFactory:
  
  @staticmethod
  def make(trainParams: SafeDict) -> T.Type[DataReader]:
    
    if trainParams['local_data_path'] is not None:
      dataLoc = 'local'
      
    elif trainParams['socrata_data_key'] is not None:
      dataLoc = 'socrata'
      
    elif trainParams['s3_data_path'] is not None:
      dataLoc = 's3'
      
    else:
      raise Error('data location cannot be inferred from trainParams')
      
    if dataLoc in ['local', 's3']:
      dataReader = LocalOrS3DataReader
      
    elif dataLoc == 'socrata':
      dataReader = SocrataDataReader
      
    else:
      raise ValueError(f'{dataLoc=} not recognized')
      
    return dataReader(trainParams)