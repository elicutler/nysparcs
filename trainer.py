
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
  
  
class TorchTrainerMixin:
  
  def _loadModel():
    pass
  
  def _initializeWeights():
    pass
  

class TorchBatchTrainer(Trainer, TorchTrainerMixin):
  
  def __init__(
    self, modelName, target, trainFromScratch, dataLoc, dataID, 
    trainBatchSize, numWorkers
  ) -> None:
    
    self.modelName = modelName
    self.trainFromScratch = trainFromScratch
    
    self.dataLoader = ThisDataReaderProcessorLoader(
      target, dataLoc, dataID, trainBatchSize, numWorkers
    )
  
  @overrides
  def train(self, epochs):

    model = self._loadModel()
    initWeights = self._initializeWeights()
    
    for e in range(epochs):
      
#       rawData = self.dataLoader.load()
#       inputData = self.dataProcessor.process()
#       updatedWeights = self._optimize()
    
  
class TorchFullTrainer(Trainer, TorchTrainerMixin):
  
  def __init__(
    self, modelName, target, trainFromScratch, dataLoc, dataID, 
    trainBatchSize, numWorkers
  ) -> None:
    
    self.modelName = modelName
    self.trainFromScratch = trainFromScratch
    
    self.dataReader = DataReaderFactory.make('full', dataLoc, dataID)
    self.dataProcessor = DataProcessor()
    self.dataLoader = ThisDataLoader(target, trainBatchSize, numWorkers)
    
  @overrides
  def train(self, epochs):
    
    df = self.dataReader.read()
    model = self._loadModel()
    initWeights = self._initializeWeights()
