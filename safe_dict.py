
import logging
import typing as T
import copy

from overrides import overrides

logger = logging.getLogger(__name__)


class SafeDict(dict):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @classmethod
  def fromNamespace(cls, namespace):
    nsDict = namespace.__dict__
    return cls(nsDict)
    

  @overrides
  def copy(self):
    return copy.deepcopy(self)
