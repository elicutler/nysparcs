
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
    # deliberate violation of Liskov substitution principle
    raise NotImplementedError('Cannot modify SafeDict')

  @overrides
  def copy(self):
    # cannot simply deepcopy(self) due to prohibition on __setitem__
    return SafeDict(self.copyUnsafeDict())
  
  def copyUnsafeDict(self):
    return {k: deepcopy(v) for k, v in self.items()}
  
  @classmethod
  def fromNamespace(cls, namespace):
    namespaceDict = namespace.__dict__
    return cls(namespaceDict)
  

