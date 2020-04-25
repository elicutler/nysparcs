
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
  def read(self) -> pd.DataFrame:
    pass


class LocalDataReader(DataReader):
  
  def __init__(self, params) -> None:
    super().__init__(params)
    
  @overrides
  def read(self) -> pd.DataFrame:
    df = pd.read_csv(
      self.params['local_data_path']
      # skiprows, nrows
      # lazy read options:
      # chunksize=n; for chunk in reader: print(chunk)
      # iterator=True -- reader.get_chunk(n)
    )


class DataReaderFactory:
  
  @staticmethod
  def make(params) -> T.Type[DataReader]:
    
    dataLoc = 'local' if params['local_data_path'] is not None else 'internet'
    
    if dataLoc == 'local':
      return LocalDataReader(params.copy())
    
    raise ValueError(f'{dataLoc=} not recognized')