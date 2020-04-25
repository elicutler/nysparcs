
import typing as T
import logging
import multiprocessing

logger = logging.getLogger(__name__)
    

def getNumCores() -> int:
  return multiprocessing.cpu_count()