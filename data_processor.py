
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)


class NYSPARCSDataProcessor:
  
  def __init__(self, dataLoc, dataID):
    
    self.dataLoc = dataLoc
    
    if self.dataLoc == 'local':
      self.dataReader = localNYSPARCSDataReader(dataID)
    else:
      self.dataReader = internetNYSPARCSDataReader(dataID)
      
  