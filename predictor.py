
import typing as T
import logging
import json
import numpy as np
import pandas as pd
import torch

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import DataLoader
from artifacts_io_handler import ArtifactsIOHandler
from model_manager import ModelManager
from data_processor import DataProcessorFactory
from torch_models import ModelArchitectureFactory
from torch_dataset import TorchDataset
from eval_no_grad import EvalNoGrad
from utils import getProcessingDevice, getNumWorkers


logger = logging.getLogger(__name__)


class Predictor(EnforceOverrides):
  
  @abstractmethod
  def __init__(self, params, artifactsMessage):
    self.params = params.copy()
    self.artifactsMessage = artifactsMessage
    
    featureCols = (
      self.artifactsMessage.meta['input_col_types']
      .drop(self.artifactsMessage.meta['target'])
      .index
    )
    self.dataProcessor = DataProcessorFactory.make(featureCols) 
    
  @abstractmethod
  def predict(self):
    pass
  
  @final
  def _parseInstances(self) -> dict:
    '''
    Parse instances from JSON file or JSON string.
    '''
    # Could use pd.from_json() instead, but this gives more flexibility
    try:
      with open(self.params['instances'], 'r') as file:
        instances = json.load(file)
        
    except FileNotFoundError:
      instances = json.loads(self.params['instances'])
      
    return instances
  
  
class PytorchPredictor(Predictor):
  
  def __init__(self, params, artifactsMessage):
    super().__init__(params, artifactsMessage)
    self.modelArchitecture = (
      ModelArchitectureFactory.make(artifactsMessage.meta['model_name'])
    )
    
  @overrides
  def predict(self) -> dict:
    
    instances = self._parseInstances()
    rawDF = pd.DataFrame.from_dict(instances, orient='index')
    self.dataProcessor.loadDF(rawDF)
    self.dataProcessor.processDF()
    processedDF = self.dataProcessor.getFeatureDF()    
    
    batchSize = processedDF.shape[0]
    numWorkers = getNumWorkers(-1)

#     XArray = np.array(sklearnProcessor.transform(processedDF).todense())
#     X = torch.from_numpy(XArray.squeeze()).float()

    torchPredDF = TorchDataset(processedDF, sklearnProcessor)

    predLoader = DataLoader(
      torchPredDF, batch_size=batchSize, num_workers=numWorkers, shuffle=False
    )
    
    device = getProcessingDevice()
    logger.info(f'Predicting on {device=} with {numWorkers=}')
    
    sklearnProcessor = self.artifactsMessage.artifacts['sklearn_processor']
    model = self.modelArchitecture(sklearnProcessor.featureNames).to(device)
    model.load_state_dict(self.artifactsMessage.artifacts['model_state_dict'])
    
    for X in predLoader:
    
      with EvalNoGrad(model):
        X = X.to(device)
        preds = (
          torch.sigmoid(model(X)) 
          if self.artifactsMessage.meta['target_type'] == 'binary'
          else model(X)
        )
        
    breakpoint()
    
    
    
    
      
class PredictorFactory:
  
  @staticmethod
  def make(params) -> T.Type[Predictor]:
    
    if (modelName := params['model_name']) is not None:
      artifactsIOHandler = ArtifactsIOHandler()
      artifactsMessage = artifactsIOHandler.load(modelName)
      
    elif params['best_model']:
      artifactsMessage = ModelManager.getBestModel(
        params['target'], params['eval_metric']
      )
      
    else:
      raise Exception('Neither model_name nor best_model truthy in params')
      
    if (modelType := artifactsMessage.meta['model_type']) == 'pytorch':
      Predictor_ = PytorchPredictor
      
    elif modelType == 'sklearn':
      Predictor_ = SKLearnPredictor
      
    else:
      raise ValueError(f'{modelType=} not recognized')
      
    return Predictor_(params, artifactsMessage)
      
      