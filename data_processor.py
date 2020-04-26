
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
    logger.info('Processing data...')
    
    df = inDF.copy()
    df.columns = self._sanitizeColNames(df.columns)
    df = self._processLOS(df)
    df = self._floatToIntCols(df)
    df = self._ynToBoolCols(df)
    df = self._objToStrCols(df)
    df = self._sanitizeStrCols(df)
    df = self._mergeCodeAndDescCols(df)
    df = self._nullInvalidContinuousCols(df)
    df = self._rmTargetOutliers(df)
    df = self._filterUnusedCols(df)
    return df
  
  def _sanitizeColNames(colNames) -> T.List[str]:
    return [self._sanitizeString(c) for c in colNames]
  
  def _sanitizeString(string) -> str:
    s = string.lower()
    s = re.sub('^\W+', '', s)
    s = re.sub('\W+$', '', s)
    s = re.sub('\W', '_', s)
    return s
  
  def _processLOS(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    losCol = df['length_of_stay']
    losCol.loc[losCol == '120 +'] = pd.NA
    losCol = losCol.astype(pd.Int64Dtype())
    losCol.loc[losCol < 0] = pd.NA
    return df
  
  def _floatToIntCols(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    floatCols = df.select_dtypes(include=['float']).columns
    
    for c in floatCols:
      try:
        df[c] = df[c].astype(pd.Int64Dtype())
        logger.info(f'Col \'{c}\' converted from float to int')
      except TypeError as e:
        logger.info(f'Failed to convert col \'{c}\' from float to int')
        
    return df
  
  def _ynToBoolCols(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    ynCols = ['abortion_edit_indicator', 'emergency_department_indicator']
    
    for c in ynCols:
      try:
        logger.info(f'Col \'{c}\' converted from object to bool')
        df[c] = df[c].map({'Y': True, 'N': False}).astype(pd.BooleanDtype())
      except TypeError as e:
        logger.info(f'Failed to convert col \'{c}\' from object to bool')
        
    return df
  
  def _objToStrCols(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    objCols = df.selec_dtypes(include=['object']).columns
    
    for c in objCols:
      try:
        logger.info(f'Col \'{c}\' converted from object to string')
        df[c] = df[c].astype(pd.StringDtype())
      except TypeError as e:
        logger.info(f'Failed to convert col \'{c}\' from object to string')
        
    return df
  
  def _sanitizeStrCols(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    strCols = df.select_dtypes(include=['string']).column
    
    for c in strCols:
      df[c] = df[c].apply(lambda x: self._sanitizeString(x))
      
    return df
  
  def _mergeCodeAndDescCols(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    colStems = ['ccs_diagnosis', 'ccs_procedure']
    
    for c in colStems:
      code = c + '_code'
      desc = c + '_description'
      codeDescMap = df[c].groupby(code)[desc].first()
      codeDescCol = (codeDescMap.index + '_' + codeDescMap.values)
      df[c] = codeDescCol
    
    return df
  
  def _nullInvalidContinuousCols(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    continuousCols = df.select_dtypes(include=['number']).columns
    
    for c in continuousCols:
      df[c].loc[df[c] < 0] = pd.NA
      
    return df
  
  def _filterTargetOutliers(self, inDF, quantile=0.99) -> pd.DataFrame:
    df = inDF.copy()
    target = self.params['target']
    targetMaxKeep = df[target].quantile(q=quantile)
    df = df[df[target] <= targetMaxKeep]
    return df
    
  def _removeUnusedCols(self, inDF) -> pd.DataFrame:
    df = inDF.copy()
    keepCols = [self.params['target']] + self.params['features']
    return df[keepCols]