
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
    df = self._mergeCodeAndDescCols(df)
    df = self._rmTargetOutliers(df)
    return df
  
  def _sanatizeColNames(colNames) -> T.List[str]:
    return [re.sub('\W', '_', c.lower()) for c in colNames]