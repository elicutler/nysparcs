
import typing as T
import logging
import multiprocessing
import re

from collections.abc import Sequence
from datetime import datetime
from pytz import timezone

logger = logging.getLogger(__name__)
    

def getNumCores() -> int:
  return multiprocessing.cpu_count()


def getMinMaxIndicesOfItemInList(item, list_, startsWith=False)  -> T.Tuple[int]:
  if startsWith:
    itemIndices = [
      i for i, el in enumerate(list_) if el.startswith(item)
    ]
  else:
    itemIndices = [i for i, el in enumerate(list_) if i == el]
  
  return itemIndices[0], itemIndices[-1]


def flattenNestedSeq(seq) -> list:
  flattened = [j for i in seq for j in i]
  if isinstance((el := flattened[0]), Sequence) and not isinstance(el, str):
    flattened = flattenNestedSeq(flattened)
  return flattened


def nowTimestampStr() -> str:
  return datetime.now(timezone('UTC')).strftime('%Y%m%d%H%M%S%f')


FIXED_SEED = 617