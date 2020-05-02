
import typing as T
import logging
import pandas as pd
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
    breakpoint()
    
    featureInputs = (
      self.df
      .drop(columns=self.params['target'])
      .iloc[idx]
      .to_frame()
      .transpose()
    )
    X =(
      self.sklearnProcessor.transform(featureInputs).todense()
    )
    print(f'DIM X: {X.shape}')
    
    y = (
      self.df[self.params['target']].iloc[idx]
    )
    
    print(f'DIM Y: {y.shape}')
    breakpoint()
    return X, y
    
  def _validateFeatures(self) -> None:
    assert (
      (self.df.drop(columns=self.params['target']).columns 
      == self.sklearnProcessor.featureInputNames)
    ).all()
    