
import typing as T
import logging
import re
import pandas as pd
import numpy as np

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from target_encoder import TargetEncoder

logger = logging.getLogger(__name__)


class SKLearnProcessor:
  
  def __init__(self, params) -> None:
    self.params = params.copy()
    
    self.trainDF = None
    self.sklearnProcessor = None

  def loadDF(self, inDF) -> None:
    self.trainDF = inDF.copy()
    
  def fit(self) -> None:
    logger.info('Fitting scikit-learn processing pipeline')
    
    if self.sklearnProcessor is not None:
      logger.warning(
        'scikit-learn processor has already been fit.'
        ' Nothing to do.'
      )
      return
    
    assert self.trainDF is not None, 'first call self.loadDF()'
    
    trainX = self.trainDF.drop(columns=[self.params['target']])
    trainY = self.trainDF[self.params['target']]
    
    numFeatureCols = trainX.select_dtypes(include=['number']).columns
    catFeatureCols = trainX.select_dtypes(include=['object']).columns
    
    catEncoder = self._getCatEncoderStrat()
    
    numPipe = Pipeline([
      ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
      ('scaler', StandardScaler())
    ])
    catPipe = Pipeline([
      ('imputer', SimpleImputer(strategy='constant', fill_value='__UNK__')),
      ('cat_encoder', catEncoder)
    ])
    pipe = ColumnTransformer([
      ('num_pipe', numPipe, numFeatureCols),
      ('cat_pipe', catPipe, catFeatureCols)
    ])
    pipe.fit(trainX, trainY)
    
    pipe.features = trainX.columns.to_list()
    self.sklearnProcessor = pipe
    
  def get(self) -> Pipeline:
    assert self.sklearnProcessor is not None, 'first call self.fit()'
    return self.sklearnProcessor
  
  def _getCatEncoderStrat(self) -> T.Union[OneHotEncoder, TargetEncoder]:
    
    if self.params['cat_encoder_strat'] == 'one_hot':
      return OneHotEncoder(handle_unknown='ignore')
    
    elif self.params['cat_encoder_strat'] == 'target':
      return TargetEncoder(priorFrac=self.params['target_encoder_prior'])

    raise ValueError(f'{cat_encoder=} not recognized')