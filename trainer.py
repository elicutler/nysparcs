
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import DataLoader
from data_reader import DataReaderFactory
from data_processor import TorchDataProcessor
from torch_dataset import TorchDataset
from utils import getNumCores

logger = logging.getLogger(__name__)
    
  
class Trainer(EnforceOverrides):
  
  def __init__(self, params) -> None:
    self.params = params
    
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
    super().__init__(params)
    
    self.dataReader = DataReaderFactory.make(params.copy())
    self.dataProcessor = TorchDataProcessor(params.copy())
    self.torchDataset = TorchDataset(params.copy())
    
    numWorkers = (
      getNumCores()-1 if self.params['num_workers'] == -1
      else self.params['num_workers']
    )
    self.dataLoader = DataLoader(
      self.torchDataset, 
      batch_size=self.params['batch_size'], 
      num_workers=numWorkers
    )
  
  @overrides
  def train(self):
    
    df = self.dataReader.readTrainRange()
    breakpoint()
    df = self.dataProcessor.process(df)
    self.torchDataset.load(df)

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
