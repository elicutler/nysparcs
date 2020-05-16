
import typing as T
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



class TargetEncoder(BaseEstimator, TransformerMixin): 
  '''
  Target mean encoding data preprocessor compatible with scikit-learn pipelines.
  Supports continuous outcomes (including binary outcomes converted to 0/1). 
  Does not support multi-class outcomes.
  -----
  
  params
    priorSize -- regularize level-specific outcome means against grand mean
      by supplying a fixed number of observations for the grand mean weight
    priorFrac -- regularize level-specific outcome means against grand mean
      by supplying a fraction of observations that gets converted to a 
      fixed number of observations for the grand mean weight
  
  public methods
    fit -- for scikit-learn processing
    transform -- for scikit-learn processing
    
  public attributes
    none
  '''
  
  def __init__(
    self, priorSize: T.Optional[int]=None, 
    priorFrac: T.Optional[float]=None
  ):
    assert not (priorSize is not None and priorFrac is not None)
    
    self.priorSize = priorSize
    self.priorFrac = priorFrac
    
  def fit(
    self, 
    X: T.Union[pd.DataFrame, np.array], 
    y: T.Union[pd.Series, np.array]
  ) -> 'TargetEncoder':
    '''
    Estimate target means for each level of each categorical
    feature (optionally regularized)
    -----
    
    params
      X -- feature array for estimating target means by feature level
      y -- target array for estimating target means by feature level
      
    returns
      self
    '''
    
    X = X.values if isinstance(X, pd.DataFrame) else X  

    if isinstance(y, pd.Series):
      y = y.values.reshape(y.shape[0], 1)  
    elif np.ndim(y) == 1:
      y = y.reshape(y.shape[0], 1)

    dataArr = np.concatenate((X, y), axis=1)
    
    grandMean = np.mean(y)
    
    levelMeans = {j: {} for j in range(X.shape[1])}
    levelCounts = {j: {} for j in range(X.shape[1])}

    for j in levelMeans.keys():
      for g in np.unique(X[:, j]):

        X_jg = dataArr[:, j] == g
        y_j = dataArr.shape[1] - 1   
        X_jg_y = dataArr[X_jg, y_j].astype(float)
        
        levelMeans[j][g] = (
          np.mean(X_jg_y) 
          if not np.isnan(np.mean(X_jg_y))
          else grandMean
        )
        levelCounts[j][g] = dataArr[X_jg].shape[0]
        
    if self.priorSize is not None or self.priorFrac is not None:
      
      levelMeansSmoothed = {j: {} for j in range(X.shape[1])}
      
      for j in levelMeans.keys():
        for g in levelMeans[j].keys():
          
          if self.priorSize is not None:
            weightSmoothed = self.priorSize / (self.priorSize + levelCounts[j][g])
          elif self.priorFrac is not None:
            priorSize = X.shape[0]*self.priorFrac
            weightSmoothed = priorSize / (priorSize + levelCounts[j][g])
            
          levelMeansSmoothed_jg = (
            (1 - weightSmoothed)*levelMeans[j][g] + weightSmoothed*grandMean
          )            
          levelMeansSmoothed[j][g] = (
            levelMeansSmoothed_jg
            if not np.isnan(levelMeansSmoothed_jg)
            else grandMean
          )   
      self.levelMeansSmoothed = levelMeansSmoothed
      
    self.levelMeans = levelMeans
    self.grandMean = grandMean 
    return self
    
  def transform(self, X: np.array) -> np.array:
    '''
    Get target means for each level of each categorical feature
    (optionally regularized)
    -----
    
    params:
      X -- feature array for which to apply target means
      
    returns
      XTransformed -- array of target means
    '''
    
    X = X.values if isinstance(X, pd.DataFrame) else X
    XTransformed = np.empty_like(X).astype(float)
    
    for j in range(XTransformed.shape[1]):
      
      if self.priorSize is not None or self.priorFrac is not None:
        meansGetter = np.vectorize(self.levelMeansSmoothed[j].get)
      else:
        meansGetter = np.vectorize(self.levelMeans[j].get)
        
      XTransformed[:, j] = meansGetter(X[:, j])
      
      # imputes for levels in validation set not seen in training set
      XTransformed[np.isnan(XTransformed[:, j]), j] = self.grandMean
        
    return XTransformed
