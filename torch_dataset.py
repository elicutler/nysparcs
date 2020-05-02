
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
  
  def __init__(self, params, inDF, sklearnProcessor) -> None:
    self.params = params.copy()
    self.df = inDF.copy()
    self.sklearnProcessor = sklearnProcessor
    
    self._validateFeatures()
    
  def __len__(self) -> int:
    return self.df.shape[0]
  
  def __getitem__(self, idx) -> T.Tuple[torch.Tensor]:
    featureInputs = (
      self.df
      .drop(columns=self.params['target'])
      .iloc[idx]
      .to_frame()
      .transpose()
    )
    X = torch.from_numpy(
      np.array(
        self.sklearnProcessor.transform(featureInputs).todense()
      ).squeeze()
    )    
    y = torch.from_numpy(
      np.array([
        self.df[self.params['target']].iloc[idx]
      ])
    )
    return X, y
    
  def _validateFeatures(self) -> None:
    assert (
      (self.df.drop(columns=self.params['target']).columns 
      == self.sklearnProcessor.featureInputNames)
    ).all()
    