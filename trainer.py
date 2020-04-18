
import logging
from abc import  abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)


class Trainer(EnforceOverrides):

  @abstractmethod
  def train(self):
    pass

  
class TorchTrainer(Trainer):
  
  @overrides
  def train(self):
    
    model = self.loadModel()
    params = self.loadParams()
    self.fit(model, params)
    
  def loadModel(self):
    return model
  
  def loadParams(self):
    return params
    
  def fit(self, model, params=None):
    
    rawData = self.loadData()
    inputData = self.processData()
    updatedParams = self.optimize()
    
    return updatedParams
  

  
