
import logging
import typing as T

from copy import deepcopy
from overrides import overrides

logger = logging.getLogger(__name__)


class SafeDict(dict):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  @overrides
  def __setitem__(self, key, value):
    raise NotImplementedError('Cannot modify SafeDict')

  @overrides
  def copy(self):
    # Cannot simply deepcopy(self), due to prohibition on __setitem__
    return SafeDict({k: deepcopy(v) for k, v in self.items()})
  
  def copyUnsafeDict(self):
    return {k: deepcopy(v) for k, v in self.items()}
  
  @classmethod
  def fromNamespace(cls, namespace):
    namespaceDict = namespace.__dict__
    return cls(namespaceDict)
  

