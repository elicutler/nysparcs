
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class DataLoaderFactory:
  
  @staticmethod
  def make(dataLoc, dataID, batchSize, numWorkers) -> T.Type[DataLoader]:
    
    if dataLoc == 'local':
      DatasetClass = LocalNYSPARCSDataset
    elif dataLoc == 'internet':
    else:
      raise ValueError(f'{dataLoc=} not recognized')
      
    return DataLoader(dataset)
    
    
class NYSPARCSDataset(Dataset, EnforceOverrides):