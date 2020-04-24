
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
  
  def __init__(self, params) -> None:
    self.params = params
    
    self.dataReader = DataReaderFactory.make(params.copy())
    breakpoint()
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
  
  @overrides
  def saveModel(self):
    pass
  
  @overrides
  def deployModel(self):
    pass
    
  
class TrainerFactory:
  
  @staticmethod
  def make(params) -> T.Type[Trainer]:
    
    modelType = 'torch' if params['torch_model'] is not None else 'sklearn'

    if modelType == 'torch':
      return TorchTrainer(params.copy())

    raise ValueError(f'{modelType=} not recognized')
