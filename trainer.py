
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from data_loader import NYSPARCSDataLoader

logger = logging.getLogger(__name__)


class TrainerFactory:
  
  @staticmethod
  def make(
    modelType, processInBatch, modelName, target, trainFromScratch, dataLoc, 
    dataID, trainBatchSize, numWorkers
  ) -> T.Type[Trainer]:
    
    trainerArgs = (
      modelName, target, trainFromScratch, dataLoc, dataId, trainBatchSize,
      numWorkers
    )
    if modelType == 'torch':
      if processInBatch:
        return TorchBatchTrainer(*trainerArgs)
      else:
        return TorchFullTrainer(*trainerArgs)
        
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
  

class TorchBatchTrainer(Trainer):
  
  def __init__(
    self, modelName, target, trainFromScratch, dataLoc, dataID, 
    trainBatchSize, numWorkers
  ) -> None:
    
    self.modelName = modelName
    self.trainFromScratch = trainFromScratch
    
    self.torchDataLoader = NYSPARCSDataProcessorAndLoader(
      target, dataLoc, dataID, trainBatchSize, numWorkers
    )
  
  @overrides
  def train(self, epochs):

    model = self._loadModel()
    initWeights = self._initializeWeights()
    
    for e in range(epochs):
      
      rawData = self.torchDataLoader.load()
      inputData = self.dataProcessor.process()
      updatedWeights = self._optimize()
    
  def _loadModel(self) -> TorchModel:
    pass
  
  def _initializeWeights(self) -> TorchWeights:
    pass


