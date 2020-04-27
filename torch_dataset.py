
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TorchDataset(Dataset):
  
  def __init__(self, inDF) -> None:
    self.df = inDF.copy()
    
  def __len__(self) -> int:
    return self.df.shape[0]
  
  def __getitem__(self, idx) -> pd.Series:
    return self.df.iloc[idx]
  
  
