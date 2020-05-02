
import typing as T
import logging
import torch_models

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import DataLoader
from data_reader import DataReaderFactory
from data_processor import DataProcessor
from sklearn_processor import SKLearnProcessor
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
    self.sklearnProcessor = SKLearnProcessor(params)
    self.torchDataset = TorchDataset(params)
  
  @overrides
  def train(self):
    
    rawDF = self.dataReader.read()
    
    self.dataProcessor.loadDF(rawDF)
    self.dataProcessor.processDF()
    trainDF, testDF = self.dataProcessor.getTrainTestDFs()
    
    self.sklearnProcessor.loadDF(trainDF)
    self.sklearnProcessor.fit()
    sklearnProcessor = self.sklearnProcessor.get()
    
    self.torchDataset.loadDF(trainDF)
    self.torchDataset.loadSKLearnProcessor(sklearnProcessor)
    self.torchDataset.validateFeatures()
    
    batchSize = self.params['batch_size']
    numWorkers = (
      getNumCores()-1 if (x := self.params['num_workers']) == -1 else x
    )
    trainLoader = DataLoader(
      trainDF, batch_size=batchSize, num_workers=numWorkers
    )
    testLoader = DataLoader(
      testDF, batch_size=batchSize, num_workers=numWorkers
    )

    model = self._loadModel(sklearnProcessor.featureNames)
    initWeights = self._initializeWeights()
    
    for e in range(epochs):
      pass
#       rawData = self.dataLoader.load()
#       inputData = self.dataProcessor.process()
#       updatedWeights = self._optimize()
  
  def _loadModel(self, featureNames):
    if (modelName := self.params['torch_model']) == 'CatEmbedNet':
      return torch_models.CatEmbedNet(featureNames)
    
  
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
