
from abc import  abstractmethod
from overrides import EnforceOverrides, overrides, final

class Trainer(EnforceOverrides):

  @abstractmethod
  def train(self):
    pass

class TorchTrainer:
  
  def __init__(
    epochs, batchSize, modelType, modelParams
  ):
    
  @overrides
  def train(self):
    
    model = self.loadModel(modelType)
    params = self.loadParams(modelParams)
    self.fit(model, params)
    
  def loadModel(self) -> model:
    pass
    
  def fit(self, model) -> model:
    
    rawData = self.loadData()
    inputData = self.processData()
    fittedModel = self.optimize()
  

  
