
import typing as T
import logging
import re
import pandas as pd
import numpy as np

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)

logger = logging.getLogger(__name__)


class DataProcessor:
  
  def __init__(self, params) -> None:
    self.params = params.copy()
    self.df = None
    self.scikitPipeline = None
    
  def loadDF(self, inDF) -> None:
    breakpoint()
    self.df = inDF.copy()
    
  def process(self) -> None:
    logger.info('Processing data...')
    
    self.df.columns = self._sanitizeColNames(self.df.columns)
    self._processLOS()
    breakpoint()
    self._floatToIntCols()
    self._ynToBoolCols()
    self._objToStrCols()
    self._sanitizeStrCols()
    self._mergeCodeAndDescCols()
    self._makePriorAuthDispo()
    self._removeUnusedCols()
    self._nullifyInvalidNumericCols()
    self._filterNumericOutliers()
    self.df.reset_index(drop=True, inplace=True)
    
  def fitScikitPipeline(self) -> None:
    numPipe = Pipeline([
      ()
    ])
    breakpoint()

    
#            catEncoder = self._getCatEncoder(catEncoderStrat)
#         numPipe = Pipeline([
#             ('numImputer', SimpleImputer(strategy=numImputerStrat)),
#             ('numScaler', StandardScaler())
#         ])
#         catPipe = Pipeline([
#             ('catImputer', SimpleImputer(strategy='most_frequent')),
#             ('catEncoder', catEncoder)
#         ])
#         numCatPipe = ColumnTransformer([
#             ('numPipe', numPipe, self.numFeatures),
#             ('catPipe', catPipe, self.catFeatures)
#         ])
#         preprocessor = FeatureUnion([
#             ('numCatPipe', numCatPipe),
#             ('missingFlagger', 
#              MissingIndicator(missing_values=missingValues, features='all')
#             )
#         ])

  
  def getTrainTestDFs(self) -> T.Tuple[pd.DataFrame]:
    trainDF = (
      self.df[self.df['train_test'] == 'train']
      .drop(columns=['train_test'])
      .reset_index(drop=True)
    )
    testDF = (
      self.df[self.df['train_test'] == 'test']
      .drop(columns=['train_test'])
      .reset_index(drop=True)
    )
    return trainDF, testDF
  
  def _sanitizeColNames(self,colNames) -> T.List[str]:
    return [self._sanitizeString(c) for c in colNames]
  
  def _sanitizeString(self, string) -> str:
    s = string.lower()
    s = re.sub('^\W+', '', s)
    s = re.sub('\W+$', '', s)
    s = re.sub('\W+', '_', s)
    return s
  
  def _processLOS(self) -> None:
    losColName = 'length_of_stay'
    losCol = self.df[losColName].copy()
    losCol.loc[losCol == '120 +'] = np.nan
    losCol = losCol.astype(np.float64).astype(pd.Int64Dtype())
    losCol.loc[losCol < 0] = pd.NA
    self.df[losColName] = losCol
  
  def _floatToIntCols(self) -> None:
    floatCols = self.df.select_dtypes(include=['float']).columns
    
    for c in floatCols:
      try:
        self.df[c] = self.df[c].astype(pd.Int64Dtype())
        logger.info(f'Col \'{c}\' converted from float to int')
      except TypeError as e:
        logger.info(f'Failed to convert col \'{c}\' from float to int')
  
  def _ynToBoolCols(self) -> None:
    ynCols = ['abortion_edit_indicator', 'emergency_department_indicator']
    
    for c in ynCols:
      try:
        logger.info(f'Col \'{c}\' converted from object to bool')
        self.df[c] = (
          self.df[c].map({'Y': True, 'N': False}).astype(pd.BooleanDtype())
        )
      except TypeError as e:
        logger.info(f'Failed to convert col \'{c}\' from object to bool')
        
  def _objToStrCols(self) -> None:
    objCols = self.df.select_dtypes(include=['object']).columns
    
    for c in objCols:
      try:
        logger.info(f'Col \'{c}\' converted from object to string')
        self.df[c] = self.df[c].astype(pd.StringDtype())
      except TypeError as e:
        logger.info(f'Failed to convert col \'{c}\' from object to string')
  
  def _sanitizeStrCols(self) -> None:
    strCols = self.df.select_dtypes(include=['string']).columns
    
    for c in strCols:
      self.df[c] = self.df[c].apply(
        lambda x: self._sanitizeString(x) if isinstance(x, str) else x
      )
      
  def _mergeCodeAndDescCols(self) -> None:
    colStems = ['ccs_diagnosis', 'ccs_procedure']
    
    for c in colStems:
      code = c + '_code'
      desc = c + '_description'
      codeDescMap = self.df.groupby(code, as_index=False)[desc].first()
      codeDescMap['merged'] = (
        codeDescMap[code].astype(str).values + '_' + codeDescMap[desc]
      )
      codeDescMergedMap = codeDescMap.set_index(code)['merged']
      self.df[c] = self.df[code].map(codeDescMergedMap)
  
  def _makePriorAuthDispo(self) -> None:
    priorAuthDispos = [ 
      'home_w__home_health_services',
      'inpatient_rehabilitation_facility',
      'psychiatric_hospital_or_unit_of_hosp',
      'skilled_nursing_home'
    ]
    self.df['prior_auth_dispo'] = (
      self.df['patient_disposition'].apply(
        lambda x: pd.NA if pd.isna(x) 
        else True if x in priorAuthDispos 
        else False
      ).astype(pd.BooleanDtype())
    )
  
  def _removeUnusedCols(self) -> None:
    keepCols = ['train_test', self.params['target']] + self.params['features']
    logger.info(
      f'Removing cols: {[c for c in self.df.columns if c not in keepCols]}'
    )
    self.df = self.df[keepCols]
  
  def _nullifyInvalidNumericCols(self) -> None:
    continuousCols = self.df.select_dtypes(include=['number']).columns
    
    for c in continuousCols:
      self.df.loc[self.df[c] < 0, c] = pd.NA
    
  def _filterNumericOutliers(self, quantile=0.99) -> None:
    numCols = [
      c for c in self.df.columns 
      if is_numeric_dtype(self.df[c]) 
      and not self.df[c].dtype == pd.BooleanDtype()
    ]
    for c in numCols:
      initNumRows = self.df.shape[0]
      
      maxKeepVal = self.df[c].quantile(q=quantile)
      self.df = self.df[self.df[c] <= maxKeepVal]
      
      rmNumRows = self.df.shape[0] - initNumRows
      logger.info(
        f'Removed {rmNumRows}/{initNumRows} rows with \'{c}\''
        f' beyond {quantile=} value.'
      )
  

    