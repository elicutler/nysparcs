
import typing as T
import logging
import pandas as pd
import json

from pathlib import Path
from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from sodapy import Socrata
from constants import S3_BUCKET

logger = logging.getLogger(__name__)


class DataReader(EnforceOverrides):
  
  def __init__(self, trainParams):
    self.trainParams = trainParams.copy()
  
  @abstractmethod
  def read(self) -> pd.DataFrame:
    pass
  
  @abstractmethod
  def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
    pass
  

class LocalOrS3DataReader(DataReader):
  
  def __init__(self, trainParams):
    super().__init__(trainParams)
    
    self.dataLoc = 'local' if self.trainParams['local_data_path'] is not None else 's3'
    
    if self.dataLoc == 'local':
      self.dataPath = Path(trainParams['local_data_path'])
    
    elif self.dataLoc == 's3':
      s3Prefix = Path(f's3://{S3_BUCKET}/nysparcs/')
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
  def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
    
    colNames = pd.read_csv(self.dataPath, nrows=0).columns
    df = pd.read_csv(
      self.dataPath, skiprows=startRow-1, nrows=numRows,
      header=None, names=colNames
    )
    return df
  

class SocrataDataReader(DataReader):
  
  def __init__(self, trainParams):
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
  def _readNumRowsFromStartRow(self, startRow, numRows) -> pd.DataFrame:
    
    socrataDataKey = self.trainParams['socrata_data_key']
    dataRecs = self.socrataConn.get(
      socrataDataKey, order=':id', offset=startRow, limit=numRows
    )
    df = pd.DataFrame.from_records(dataRecs)
    return df
  
  def _establishSocrataConn(self) -> Socrata:
    with open('.config/secrets.json', 'r') as f:
      appToken = json.load(f)['socrata']['app_token']
    return Socrata('health.data.ny.gov', appToken)
                

class DataReaderFactory:
  
  @staticmethod
  def make(trainParams) -> T.Type[DataReader]:
    
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