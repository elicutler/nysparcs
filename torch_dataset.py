
import typing as T
import logging
import pandas as pd

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TorchDataset(Dataset):
  
  def __init__(self, inDF, target) -> None:
    self.df = inDF.copy()
    self.target = target
    
  def __len__(self) -> int:
    return self.df.shape[0]
  
  def __getitem__(self, idx) -> T.Tuple[pd.Series, bool]:
    return (
      self.df.drop(columns=[self.target]).iloc[idx],
      self.df.iloc[idx][self.target]
    )
  
  
