
import typing as T
import logging
import multiprocessing
import re

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