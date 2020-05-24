
import typing as T
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch_models
import pandas as pd
import numpy as np

from collections import OrderedDict
from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, median_absolute_error
from data_reader import DataReaderFactory
from data_processor import DataProcessor
from sklearn_processor import SKLearnProcessor
from artifacts_io_handler import ArtifactsIOHandler
from torch_dataset import TorchDataset
from eval_no_grad import EvalNoGrad
from utils import getNumCores
from constants import FIXED_SEED
from sklearn_pipelines import SKLearnPipelineMaker

logger = logging.getLogger(__name__)
    
  
class Trainer(EnforceOverrides):
  
  @abstractmethod
  def __init__(self, params):
    self.params = params.copy()
    self.dataReader = DataReaderFactory.make(params)
    self.dataProcessor = DataProcessor(params)
    self.artifactsIOHandler = ArtifactsIOHandler(params)
    
    self.inputColTypes = None
    self.valPerformanceMetrics = None
    
  @abstractmethod
  def train(self):
    pass
  
  @abstractmethod
  def saveModel(self):
    pass

  @final
  def _calcPerformanceMetrics(self, actuals, preds) -> T.Dict[str, float]:
    
    if (targetType := self.params['target_type']) == 'binary':
      perfCalculator = self._calcBinaryPerformanceMetrics
    
    elif targetType == 'regression':
      perfCalculator = self._calcRegressionPerformanceMetrics
    
    else:
      raise ValueError(f'{targetType=} not recognized')
      
    metrics = perfCalculator(actuals, preds)
    logger.info(f'Performance metrics:\n{metrics}')
    return metrics
      
  @final
  def _calcBinaryPerformanceMetrics(self, actuals, preds) -> T.Dict[str, float]:
    metrics = {
      'confusion_matrix': confusion_matrix(actuals, preds >= 0.5),
      'roc_auc': roc_auc_score(actuals, preds),
      'pr_auc': average_precision_score(actuals, preds),
      'prop_trues': np.mean(actuals),
      'nobs': preds.shape[0]
    }
    return metrics
  
  @final
  def _calcRegressionPerformanceMetrics(self, actuals, preds) -> T.Dict[str, float]:
    metrics = {
      'mean_abs_err': mean_absolute_error(actuals, preds),
      'med_abs_err': median_absolute_error(actuals, preds),
      'mean_y': np.mean(actuals),
      'med_y': np.median(y),
      'nobs': preds.shape[0]
    }
    return metrics
  

class TorchTrainer(Trainer):

  @overrides
  def __init__(self, params):
    super().__init__(params)
    self.sklearnProcessor = SKLearnProcessor(params)
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
    numWorkers = (
      getNumCores()-1 if (x := self.params['num_workers']) == -1 else x
    )
    logger.info(f'Running on {numWorkers} cores')
    
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
    
    finalEpochValPreds = torch.empty(0, dtype=torch.float)
    finalEpochValActuals = torch.empty(0, dtype=torch.float)
    
    torch.manual_seed(FIXED_SEED)
    
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
        runningEpochTrainNobs += y.size()[0]
        
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
        runningEpochValNobs += y.size()[0]
        
        if i == numEpochs: 
          finalEpochValPreds = torch.cat(
            [finalEpochValPreds, torch.sigmoid(preds.squeeze())]
          )
          finalEpochValActuals = torch.cat([finalEpochValActuals, y.squeeze()])
          
      avgEpochValLoss = runningEpochValLoss / runningEpochValNobs
      allEpochValLosses.append(avgEpochValLoss)
      logger.info(f'{avgEpochValLoss=}')
      
      if i == numEpochs: 
        self.valPerformanceMetrics = self._calcPerformanceMetrics(
          finalEpochValActuals.numpy(), finalEpochValPreds.numpy()
        )
        
    logger.info('Training complete')

  @overrides
  def saveModel(self) -> None:

    meta = {
      'model_type': 'torch',
      'target': self.params['target'],
      'val_range': self.params['val_range'],
      'val_perf_metrics': self.valPerformanceMetrics,
      'input_col_types': self.inputColTypes
    }
    artifacts = {
      'sklearn_processor': self.sklearnProcessor.get(),
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
    }
    message = self.artifactsIOHandler.Message(meta, artifacts)
    self.artifactsIOHandler.save(message)

  def _loadModel(self, featureNames) -> T.Type[nn.Module]:
    
    if (modelName := self.params['pytorch_model']) == 'CatEmbedNet':
      modelClass = torch_models.CatEmbedNet
      
    else:
      raise ValueError(f'{modelClass=} not recognized')
      
    return modelClass(featureNames)
  
  def _loadStateDicts(self) -> None:
    
    if (checkpoint := self.artifactsIOHandler.loadTorch()) is not None:
      
      self._validateInputColumns(checkpoint['input_col_types'])
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      
    else:
      logger.warning('No state dict loaded')
      
  def _makeLossCriterion(self) -> None:
    
    if (targetType := self.params['target_type']) == 'binary':
      lossType = nn.BCEWithLogitsLoss
      
    elif targetType == 'regression':
      lossType = nn.L1Loss
      
    else:
      raise ValueError(f'{targetType=} not recognized')
      
    return lossType(reduction='sum')
  
  def _validateInputColumns(self, loadedInputColTypes) -> None:
    assert (self.inputColTypes.index == loadedInputColTypes.index).all(), (
      f'Input column names do not match:\n{self.inputColTypes=}'
      f'\n{loadedInputColTypes=}'
    )
    assert (self.inputColTypes.values == loadedInputColTypes.values).all(), (
      f'Input column data types do not match:\n{self.inputColTypes=}'
      f'\n{loadedInputColTypes=}'
    )
    
    
class SKLearnTrainer(Trainer):
  
  def __init__(self, params):
    super().__init__(params)
    self.sklearnPipelineMaker = SKLearnPipelineMaker(params)
    
  @overrides
  def train(self) -> None:
    
    rawDF = self.dataReader.read()
    
    self.dataProcessor.loadDF(rawDF)
    self.dataProcessor.processDF()
    trainDF, valDF = self.dataProcessor.getTrainValDFs()
    
    self.inputColTypes = trainDF.dtypes
    
    self.sklearnPipelineMaker.loadInputColTypes(self.inputColTypes)
    pipeline = self.sklearnPipelineMaker.makePipeline()
    
    trainX, trainY = self._splitXY(trainDF)
    logger.info(
      f'Running hyperparameter search for {self.params["n_iter"]} iterations'
    )
    pipeline.fit(trainX, trainY)
    logger.info('Training complete')
    
    valX, valY = self._splitXY(valDF)
    valPreds = pipeline.predict(valX)
    
    self.pipeline = pipeline
    self.valPerformanceMetrics = self._calcPerformanceMetrics(valY, valPreds)
    
  @overrides 
  def saveModel(self) -> None:
    artifacts = {
      'target': self.params['target'],
      'val_range': self.params['val_range'],
      'input_col_types': self.inputColTypes,
      'pipeline': self.pipeline,
      'val_perf_metrics': self.valPerformanceMetrics
    }
    self.artifactsIOHandler.saveSKLearn(artifacts)
  
  def _splitXY(self, inDF) -> T.Tuple[pd.DataFrame, pd.Series]:
    target = self.params['target']
    df = inDF.copy()
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
    
  
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
