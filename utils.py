
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
    
  
def parseSecrets() -> T.Dict[str, str]:
  config = ConfigParser()
  config.read(SECRETS_INI)
  secrets = {
    k: v for s in config.sections() 
    for k, v in config['nysparcs'].items()
  }
  return secrets


def initializeSession() -> boto3.session:
  secrets = parseSecrets()
  session = boto3.Session(
    aws_access_key_id=secrets['aws_access_key_id'],
    aws_secret_access_key=secrets['aws_secret_access_key'],
    region_name=secrets['region_name']
  )
  return session
  

def getNumWorkers(numWorkers) -> int:
  return multiprocessing.cpu_count() - 1 if numWorkers == -1 else numWorkers


def getProcessingDevice() -> str:
  return 'cuda' if torch.cuda.is_available() else 'cpu'


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




