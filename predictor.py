
import typing as T
import logging
import json
import pandas as pd

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from artifacts_io_handler import ArtifactsIOHandler
from data_processor import DataProcessorFactory
from model_manager import ModelManager

logger = logging.getLogger(__name__)


class Predictor(EnforceOverrides):
  
  @abstractmethod
  def __init__(self, params, artifactsMessage):
    self.params = params.copy()
    self.artifactsMessage = artifactsMessage
    self.dataProcessor = DataProcessorFactory.make('train', params) #change to predict after debugging
    
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
    
  @overrides
  def predict(self) -> dict:
    
    instances = self._parseInstances()
    rawDF = pd.DataFrame.from_dict(instances, orient='index')
    self.dataProcessor.loadDF(rawDF)
    breakpoint()
    self.dataProcessor.processDF()
    processedDF = self.dataProcessor.getProcessedDF()    
    
      
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
      
      