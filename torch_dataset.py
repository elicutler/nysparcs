
import typing as T
import logging
import pandas as pd
import numpy as np
import torch

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TorchDataset(Dataset):
  
  def __init__(self, inDF, sklearnProcessor, target=None):
    self.df = inDF.copy()
    self.sklearnProcessor = sklearnProcessor
    self.target = target
    
    self._validateFeatures()
    
  def __len__(self) -> int:
    return self.df.shape[0]
  
  def __getitem__(self, idx) -> T.Union[T.Tuple[torch.Tensor], torch.Tensor]:

    inputDF = self._makeInputDF()
    featureInputs = df.iloc[idx].to_frame().transpose()
    
    XMatrix = self.sklearnProcessor.transform(featureInputs).todense()
    XArray = np.array(XMatrix).squeeze()
    X = torch.from_numpy(XArray).float()
    
    if self.target is not None:
      yArray = np.array(self.df[self.target].iloc[idx])
      y = torch.from_numpy(yArray).float()
      
    return X, y if self.target is not None else X
    
  def _validateFeatures(self) -> None:
    inputDF = self._makeInputDF()
    assert (
      (inputDF.columns == self.sklearnProcessor.featureInputNames).all()
    )
    
  def _makeInputDF(self) -> pd.DataFrame:
    inputDF = (
      self.df.drop(columns=self.target) 
      if self.target is not None 
      else self.df.copy()
    )
    return inputDF
    