
import typing as T
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import DataLoader
from data_reader import DataReaderFactory
from data_processor import DataProcessor
from torch_dataset import TorchDataset
from utils import getNumCores

logger = logging.getLogger(__name__)
    
  
class Trainer(EnforceOverrides):
  
  def __init__(self, params) -> None:
    self.params = params.copy()
    
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
    
    self.dataReader = DataReaderFactory.make(params)
    self.dataProcessor = DataProcessor(params)
  
  @overrides
  def train(self):
    
    rawDF = self.dataReader.read()
    self.dataProcessor.loadDF(rawDF)
    self.dataProcessor.process()
    trainDF, testDF = self.dataProcessor.getDF()
    
    trainDataset = TorchDataset(trainDF, self.params['target'])
    trainLoader = DataLoader(
      trainDataset, batch_size=self.params['batch_size'],
      num_workers=(
        getNumCores()-1 if self.params['num_workers'] == -1
        else self.params['num_workers']
      )
    )
    
    testDataset = TorchDatset(testDF, self.params['target'])
    testLoader = DataLoader(
      testDataset, batch_size=len(testDataset),
      num_workers=(
        getNumCores()-1 if self.params['num_workers'] == -1
        else self.params['num_workers']
      )
    )
    
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
      return TorchTrainer(params)

    raise ValueError(f'{modelType=} not recognized')
