
import typing as T
import logging
import json

from overrides import EnforceOverrides, overrides, final
from model_manager import ModelManagerFactory
from data_processor import DataProcessor

logger = logging.getLogger(__name__)


class Predictor(EnforceOverrides):
  
  @abstractmethod
  def __init__(self, params):
    self.params = params.copy()
    
  @abstractmethod
  def predict(self):
    pass
  
  @final
  def _parseInstances(self) -> dict:
    '''
    Parse instances from JSON file or JSON string.
    '''
    try:
      with open(self.params['instances'], 'r') as file:
        instances = json.load(file)
        
    except FileNotFoundError:
      instances = json.loads(self.params['instances'])
      
    return instances
  
  
class PytorchPredictor(Predictor):
  
  def __init__(self, params):
    super().__init__(params)
    
  @overrides
  def predict(self) -> dict:
    
    instances = self._parseInstances()
    rawDF = 
    
    
    modelManager = ModelManagerFactory.make(params)

    if (modelName := self.params['model_name']) is not None:
      model = modelManager.loadModel(modelName)

    elif self.params['best_model'] is not None:
      model = modelManager.loadBestModel(
        params['target'], params['eval_metric']
      )
      