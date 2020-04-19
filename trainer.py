
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from data_loader import NYSPARCSDataLoader

logger = logging.getLogger(__name__)


class TrainerFactory:
  
  @staticmethod
  def make(
   modelType, modelName, dataLoc, dataID, batchSize, trainFromScratch=False
  ) -> T.Type[Trainer]:
    
    if modelType == 'torch':
      return TorchTrainer(
        modelName, dataLoc, dataID, batchSize, trainFromScratch
      )
    else:
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
    self, modelName, dataLoc, dataID, batchSize, numWorkers, trainFromScratch=False
  ) -> None:
    
    self.modelName = modelName
    self.trainFromScratch = trainFromScratch
    self.dataLoader = NYSPARCSDataLoader(dataLoc, dataID, batchSize, numWorkers)
  
  @overrides
  def train(self, epochs):

    model = self._loadModel()
    initWeights = self._initializeWeights()
    
    for e in range(epochs):
      
      rawData = self.dataLoader.load(batchSize)
      inputData = self.dataProcessor.process()
      updatedWeights = self._optimize()
    
  def _loadModel(self) -> TorchModel:
    pass
  
  def _initializeWeights(self) -> TorchWeights:
    pass