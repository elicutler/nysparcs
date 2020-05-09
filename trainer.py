
import typing as T
import logging
import pathlib
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch_models

from collections import OrderedDict
from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import DataLoader
from data_reader import DataReaderFactory
from data_processor import DataProcessor
from sklearn_processor import SKLearnProcessor
from torch_dataset import TorchDataset
from eval_no_grad import EvalNoGrad
from utils import getNumCores, nowTimestampStr

logger = logging.getLogger(__name__)
    
  
class Trainer(EnforceOverrides):
  
  @abstractmethod
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

  @overrides
  def __init__(self, params) -> None:
    super().__init__(params)
    
    self.dataReader = DataReaderFactory.make(params)
    self.dataProcessor = DataProcessor(params)
    self.sklearnProcessor = SKLearnProcessor(params)
    
    self.inputColTypes = None
    self.model = None
    self.optimizer = None
  
  @overrides
  def train(self):
    
    rawDF = self.dataReader.read()
    
    self.dataProcessor.loadDF(rawDF)
    self.dataProcessor.processDF()
    trainDF, valDF = self.dataProcessor.getTrainValDFs()
    
    self.inputColTypes = trainDF.dtypes
    
    self.sklearnProcessor.loadDF(trainDF)
    self.sklearnProcessor.fit()
    sklearnProcessor = self.sklearnProcessor.get()
    
    torchTrainDF = TorchDataset(self.params, trainDF, sklearnProcessor)
    torchValDF = TorchDataset(self.params, valDF, sklearnProcessor)
    
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Training on {device=}')
    
    self.model = self._loadModel(sklearnProcessor.featureNames).to(device)
    self.optimizer = optim.Adam(self.model.parameters())
    
    if (
      self.params['load_latest_state_dict'] 
      or self.params['load_state_dict'] is not None
    ):
      self._loadStateDicts()
      
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
        
        self.optimizer.zero_grad()
        preds = self.model(X)
        loss = lossCriterion(preds, y)
        loss.backward()
        self.optimizer.step()
        
        runningEpochTrainLoss += loss.item()
        runningEpochTrainNobs += y.shape[0]
        
      avgEpochTrainLoss = runningEpochTrainLoss / runningEpochTrainNobs
      allEpochTrainLosses.append(avgEpochTrainLoss)
      logger.info(f'{avgEpochTrainLoss=}')
      
      runningEpochValLoss = 0.
      runningEpochValNobs = 0
      
      for X, y in valLoader:
        
        with EvalNoGrad(self.model):
          X = X.to(device)
          y = y.to(device)
          
          preds = self.model(X)
          loss = lossCriterion(preds, y)
          
        runningEpochValLoss += loss.item()
        runningEpochValNobs += y.shape[0]
          
      avgEpochValLoss = runningEpochValLoss / runningEpochValNobs
      allEpochValLosses.append(avgEpochValLoss)
      logger.info(f'{avgEpochValLoss=}')
      
    logger.info('Training complete')

  @overrides
  def saveModel(self) -> None:
    pytorchArtifactsDir = pathlib.Path('artifacts/pytorch/')
    modelName = self.params['pytorch_model']
    
    if modelName not in pathlib.os.listdir(pytorchArtifactsDir):
      pathlib.os.mkdir(pytorchArtifactsDir/modelName)
      
    thisModelDir = pytorchArtifactsDir/modelName
    modelPath = thisModelDir/f'{modelName}_{nowTimestampStr()}.pt'
    artifacts = {
      'input_col_types': self.inputColTypes,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict()
    }
    torch.save(artifacts, modelPath)
    logger.info(f'Saving model artifacts to {modelPath}')
  
  @overrides
  def deployModel(self):
    pass
  
  def _loadModel(self, featureNames) -> T.Type[nn.Module]:
    
    if (modelName := self.params['pytorch_model']) == 'CatEmbedNet':
      modelClass = torch_models.CatEmbedNet
      
    else:
      raise ValueError(f'{modelClass=} not recognized')
      
    return modelClass(featureNames)
  
  def _loadStateDicts(self) -> None:
    pytorchArtifactsDir = pathlib.Path('artifacts/pytorch/')
    
    if self.params['load_latest_state_dict']:
      
      if (
        (modelName := self.params['pytorch_model'])
        not in (pytorchArtifactsDirContents := pathlib.os.listdir(pytorchArtifactsDir))
      ):
        logger.warning(f'No previous state dicts found for {modelName=}')
        return
      
      else:
        artifacts = [
          a for a in pathlib.os.listdir(pytorchArtifactsDir/modelName)
          if a.startswith(modelName)
        ]
        
        if len(artifacts) == 0:
          logger.warning(f'No previous state dicts found for {modelName=}')
          return
        else:
          artifacts.sort(reverse=True)
        
    elif (targetModel := self.params['load_state_dict']) is not None:
      artifacts = [
        a for a in pathlib.os.listdir(pytorchArtifactsDir/modelName)
        if (
          re.sub('\.pt|\.pth', '', targetModel) 
          == re.sub('\.pt|\.pth', '', a)
        )
      ]
      assert len(artifacts) > 0, f'{targetModel=} not found'
      assert len(artifacts) < 2, f'multiple artifacts found for {targetModel=}'
      
    else:
      raise Exception(
        'Invalid combination of load_latest_state_dict and load_state_dict args'
      )
      
    artifactsPath = pytorchArtifactsDir/modelName/artifacts[0]
    logger.info(f'Loading model and optimizer state dicts from {artifactsPath}')
    
    checkpoint = torch.load(artifactsPath)
    self._validateInputColumns(checkpoint['input_col_types'])
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      
  def _makeLossCriterion(self) -> None:
    
    if (target := self.params['target']) == 'prior_auth_dispo':
      lossType = nn.BCEWithLogitsLoss
      
    elif target == 'los':
      lossType = nn.L1Loss
      
    else:
      raise ValueErro(f'{target=} not recognized')
      
    return lossType(reduction='sum')
  
  def _validateInputColumns(self, loadedInputColTypes) -> None:
    assert (self.inputColTypes.index == loadedInputColTypes.index).all(), (
      f'Input column names do not match:\n{self.inputColTypes.columns=}'
      f'\n{loadedInputColTypes.columns=}'
    )
    assert (self.inputColTypes.values == loadedInputColTypes.values).all(), (
      f'Input column data types do not match:\n{self.inputColTypes.columns}'
      '\n{loadedInputColTypes.columns=}'
    )
    
    
class SKLearnTrainer(Trainer):
  
  def __init__(self, params) -> None:
    super().__init__(params)
    
    self.dataReader = DataReaderFactory.make(params)
    self.dataProcessor = DataProcessor(params)
    
  @overrides
  def train(self) -> None:
    
    rawDF = self.dataReader.read()
    
    self.dataProcessor.loadDF(rawDF)
    self.dataProcessor.processDF()
    trainDF, valDF = self.dataProcessor.getTrainValDFs()
    
    self.inputColTypes = trainDF.dtypes
    
    breakpoint()
    
  @overrides 
  def saveModel(self) -> None:
    pass
  
  @overrides
  def deployModel(self) -> None:
    pass
  
  
class TrainerFactory:
  
  @staticmethod
  def make(params) -> T.Type[Trainer]:
    
    modelType = 'pytorch' if params['pytorch_model'] is not None else 'sklearn'

    if modelType == 'pytorch':
      trainer = TorchTrainer
    
    elif modelType == 'sklearn':
      trainer = SKLearnTrainer
    
    else:
      raise ValueError(f'{modelType=} not recognized')
      
    return trainer(params)
