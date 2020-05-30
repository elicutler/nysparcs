
import typing as T
import logging
import multiprocessing
import re
import torch
import boto3

from configparser import ConfigParser
from collections.abc import Sequence
from datetime import datetime
from pytz import timezone
from constants import SECRETS_INI

logger = logging.getLogger(__name__)
    
  
def parseSecrets() -> T.Mapping[str, str]:
  config = ConfigParser()
  config.read(SECRETS_INI)
  secrets = {
    k: v for s in config.sections() 
    for k, v in config['nysparcs'].items()
  }
  return secrets
  

def getNumWorkers(numWorkers: T.Optional[int]) -> int:
  return multiprocessing.cpu_count() - 1 if numWorkers == -1 else numWorkers


def getProcessingDevice() -> str:
  return 'cuda' if torch.cuda.is_available() else 'cpu'


def getMinMaxIndicesOfItemInSeq(
  item: str, seq: T.Sequence, startsWith: bool=False
)  -> T.Sequence[int]:
  
  if startsWith:
    itemIndices = [
      i for i, el in enumerate(seq) if el.startswith(item)
    ]
  else:
    itemIndices = [i for i, el in enumerate(seq) if i == el]
  
  return itemIndices[0], itemIndices[-1]


def flattenNestedSeq(seq: T.Sequence) -> list:
  flattened = [j for i in seq for j in i]
  if isinstance((el := flattened[0]), Sequence) and not isinstance(el, str):
    flattened = flattenNestedSeq(flattened)
  return flattened


def nowTimestampStr() -> str:
  return datetime.now(timezone('UTC')).strftime('%Y%m%d%H%M%S%f')




