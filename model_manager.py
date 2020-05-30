
import typing as T
import logging 
import argparse
import pathlib
import re
import torch
import pickle

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from overrides import EnforceOverrides, overrides, final
from sagemaker.s3 import S3Downloader
from artifacts_io_handler import ArtifactsIOHandler, ArtifactsMessage
from utils import initializeSession

logger = logging.getLogger(__name__)


class ModelManager:
  
  def __init__(self):
    self.artifactsIOHandler = ArtifactsIOHandler()    
    self.session = initializeSession()
  
  def getBestModel(self, target, evalMetric) -> ArtifactsMessage:
    
    s3ArtifactsTargetPath = (
      self.artifactsIOHandler.s3ArtifactsPath + f'{target}/'
    )
    s3ModelPaths = (
      S3Downloader.list(s3ArtifactsTargetPath, session=self.session)
    )
    modelNames = [Path(model).stem for model in s3ModelPaths]
    
    with ThreadPoolExecutor() as executor:
      models = list(executor.map(self.artifactsIOHandler.load, modelNames))
      
    bestModel = models[0]
    for mod in models[1:]:
      
      bestScore = bestModel.meta['val_perf_metrics'][evalMetric]
      challengerScore = mod.meta['val_perf_metrics'][evalMetric]
      
      if evalMetric in ['pr_auc', 'roc_auc']:
        if challengerScore > bestScore:
          bestModel = mod
          
      elif evalMetric in ['mean_absolute_error', 'mean_squared_error']:
        if challengerScore < bestScore:
          bestModel = mod
          
      else:
        raise ValueError(f'{evalMetric=} not recognized')
        
    return bestModel
      

    