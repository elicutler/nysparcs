
import typing as T
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch_models

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from collections import OrderedDict
from torch.utils.data import DataLoader
from data_reader import DataReaderFactory
from data_processor import DataProcessor
from sklearn_processor import SKLearnProcessor
from torch_dataset import TorchDataset
from eval_no_grad import EvalNoGrad
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
  
  @overrides
  def train(self):
    
    rawDF = self.dataReader.read()
    
    self.dataProcessor.loadDF(rawDF)
    self.dataProcessor.processDF()
    trainDF, valDF, testDF = self.dataProcessor.getTrainValTestDFs()
    
    self.sklearnProcessor.loadDF(trainDF)
    self.sklearnProcessor.fit()
    sklearnProcessor = self.sklearnProcessor.get()
    
    torchTrainDF = TorchDataset(self.params, trainDF, sklearnProcessor)
    torchValDF = TorchDataset(self.params, valDF, sklearnProcessor)
    torchTestDF = TorchDataset(self.params, testDF, sklearnProcessor)
    
    
    batchSize = self.params['batch_size']
#     numWorkers = (
#       getNumCores()-1 if (x := self.params['num_workers']) == -1 else x
#     )
    numWorkers = 0 # DataLoader error with multiprocessing
    trainLoader = DataLoader(
      torchTrainDF, batch_size=batchSize, num_workers=numWorkers, shuffle=False
    )
    valLoader = DataLoader(
      torchValDF, batch_size=batchSize, num_workers=numWorkers, shuffle=False
    )
    testLoader = DataLoader(
      torchTestDF, batch_size=batchSize, num_workers=numWorkers, shuffle=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Training on {device=}')
    
    model = self._loadModel(sklearnProcessor.featureNames).to(device)
    breakpoint()
    
    weights = self._loadWeights() or model.parameters()
    optimizer = optim.Adam(weights)
    lossCriterion = self._makeLossCriterion()
    
    allEpochTrainLosses = []
    allEpochValLosses = []
    
    for i in range(1, (numEpochs := self.params['epochs'])+1):
      logger.info(f'Training epoch {i}/{numEpochs}')
      
      runningEpochTrainLoss = 0.    
      runningEpochTrainNobs = 0
      
      for j, (X, y) in enumerate(trainLoader, 1):
        
        if j % 100 == 0:
          logger.info(f'Training batch {j} of epoch {i}')
          
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        preds = model(X)
        loss = lossCriterion(preds, y)
        loss.backward()
        optimizer.step()
        
        runningEpochTrainLoss += loss.item()
        runningEpochTrainNobs += y.shape[0]
        
      avgEpochTrainLoss = runningEpochTrainLoss / runningEpochTrainNobs
      allEpochTrainLosses.append(avgEpochTrainLoss)
      logger.info(f'{avgEpochTrainLoss=}')
      
      runningEpochValLoss = 0.
      runningEpochValNobs = 0
      
      for X, y in valLoader:
        
        with EvalNoGrad(model):
          X = X.to(device)
          y = y.to(device)
          
          preds = model(X)
          loss = lossCriterion(preds, y)
          
        runningEpochValLoss += loss.item()
        runningEpochValNobs += y.shape[0]
          
      avgEpochValLoss = runningEpochValLoss / runningEpochValNobs
      allEpochValLosses.append(avgEpochValLoss)
      logger.info(f'{avgEpochValLoss=}')
      
    logger.info('Training complete')

  def _loadModel(self, featureNames) -> T.Type[nn.Module]:
    
    if (modelName := self.params['torch_model']) == 'CatEmbedNet':
      modelClass = torch_models.CatEmbedNet
      
    else:
      raise ValueError(f'{modelClass=} not recognized')
      
    return modelClass(featureNames)
  
  def _loadWeights(self) -> T.Union[None, OrderedDict]:
    if (weightsPath := self.params['init_weights']) is None:
      return
    
  
  def _makeLossCriterion(self) -> None:
    
    if (target := self.params['target']) == 'prior_auth_dispo':
      lossType = nn.BCEWithLogits
      
    elif target == 'los':
      lossType = nn.L1Loss
      
    else:
      raise ValueErro(f'{target=} not recognized')
      
    return lossType(reduction='sum')
  
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
