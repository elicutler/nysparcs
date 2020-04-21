
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from data_reader import DataReaderFactory
from torch_dataset import TorchDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
    
  
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
    self, modelName, target, trainFromScratch, dataLoc, dataID, 
    batchSize, numWorkers
  ) -> None:
    
    self.modelName = modelName
    self.trainFromScratch = trainFromScratch
    
    self.dataReader = DataReaderFactory.make(dataLoc, dataID)
    self.dataProcessor = DataProcessor()
    self.dataset = TorchDataset(target)
    self.dataLoader = DataLoader(
      self.dataset, batch_size=batchSize, num_workers=numWorkers
    )
  
  @overrides
  def train(self, epochs):

    model = self._loadModel()
    initWeights = self._initializeWeights()
    
    for e in range(epochs):
      pass
#       rawData = self.dataLoader.load()
#       inputData = self.dataProcessor.process()
#       updatedWeights = self._optimize()
  
  def _loadModel(self):
    pass
  
  def _initializeWeights(self):
    pass
    
  
class TrainerFactory:
  
  @staticmethod
  def make(
    modelType, modelName, target, trainFromScratch, dataLoc, 
    dataID, batchSize, numWorkers
  ) -> T.Type[Trainer]:
    
    trainerArgs = (
      modelName, target, trainFromScratch, dataLoc, dataId, batchSize,
      numWorkers
    )
    if modelType == 'torch':
      return TorchTrainer(*trainerArgs)
        
    raise ValueError(f'{modelType=} not recognized')
