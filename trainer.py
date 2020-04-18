
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from data_loader import DataLoaderFactory

logger = logging.getLogger(__name__)


class TrainerFactory:
  
  @staticmethod
  def make(
   modelType, model, dataLoc, dataID, trainFromScratch=False
  ) -> T.Type[Trainer]:
    
    if modelType == 'torch':
      return TorchTrainer(
        model, dataLoc, dataID, trainFromScratch
      )
    
    raise ValueError(f'{modelType=} not recognized')
    

class Trainer(EnforceOverrides):

  @abstractmethod
  def train(self):
    pass
  
  @abstractmethod
  def saveModel(self):
    pass
  
  @abstractmethod
  def deployModel(self):
    pass
  
  
class TorchTrainer(Trainer):
  
  def __init__(
    self, model, dataLoc, dataID, trainFromScratch
  ) -> None:
    
    self.model = self._loadModel(model)
    self.dataLoader = DataLoaderFactory.make(dataLoc, dataID)
    
    if not trainFromScratch:
      self.weights = self._loadWeights()
      
  def _loadModel(self) -> TorchModel:
    pass
  
  def _loadWeights(self) -> TorchWeights:
    pass
  
  @overrides
  def train(self, epochs, batchsize):

    for e in range(epochs):
      rawData = self.dataLoader.load(batchSize)
      inputData = self.dataProcessor.process()
      updatedWeights = self._optimize()
    

  
