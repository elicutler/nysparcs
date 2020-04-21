
import typing as T
import logging
import argparse
import json

logger = logging.getLogger(__name__)
    
  
def parseArgsFromRunID(runID, runConfigLoc) -> argparse.Namespace:
  
  with open(runConfigLoc, 'r') as file:
    runConfigDict = json.load(file)
    
  runConfig = runConfigDict[runID]
  namespace = argparse.Namespace()
  
  for k, v in runConfig.items():
    setattr(namespace, k, v)

  return namespace