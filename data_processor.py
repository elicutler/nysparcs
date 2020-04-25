
import typing as T
import logging
import re
import pandas as pd

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)


class DataProcessor:
  
  def __init__(self, params) -> None:
    self.params = params.copy()
    
  def process(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    df.columns = self._sanatizeColNames(df.columns)
    df = self._floatToIntCols(df)
    df = self._ynToBoolCols(df)
    df = self._objToStrCols(df)
    df = self._sanatizeStrCols(df)
    df = self._mergeCodeAndDescCols(df)
    df = self._rmTargetOutliers(df)
    df = self._removeUnusedCols(df)
    return df
  
  def _sanatizeColNames(colNames) -> T.List[str]:
    return [self._sanatizeString(c) for c in colNames]
  
  def _sanatizeString(string) -> str:
    s = string.lower()
    s = re.sub('\W', '_', s)
    return s
  
  def _floatToIntCols(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    floatCols = df.select_dtypes(include=['float'])
    
    for c in floatCols:
      try:
        df[c] = df[c].astype(pd.Int64Dtype())
        logger.info(f'Column \'{c}\' converted from float to int')
      except TypeError as e:
        logger.info(f'Attempt to convert column \'{c}\' from float to int failed')
        
    return df
    
  
  def _removeUnusedCols(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    keepCols = self.params['target'] + self.params['features']
    return df[keepCols]