
import typing as T
import logging
import pandas as pd
import numpy as np

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TorchDataset(Dataset):
  
  def __init__(self, params) -> None:
    self.params = params.copy()
    
  def loadDF(self, inDF) -> None:
    self.df = inDF.copy()
    
  def loadSKLearnProcessor(self, sklearnProcessor) -> None:
    self.sklearnProcessor = sklearnProcessor
    
  def validateFeatures(self) -> None:
    assert (
      (self.df.drop(columns=self.params['target']).columns 
      == self.sklearnProcessor.features)
    ).all()
    
  def __len__(self) -> int:
    return self.df.shape[0]
  
  def __getitem__(self, idx) -> T.Tuple[np.matrix, np.float64]:
    
    featureInputs = (
      self.df
      .drop(columns=self.params['target'])
      .iloc[idx]
      .to_frame()
      .transpose()
    )
    X = self.sklearnProcessor.transform(featureInputs).todense()
    y = self.df[self.params['target']].iloc[idx]
    
    return X, y
    
