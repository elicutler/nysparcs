
import typing as T
import logging
import torch
import torch.nn as nn
import torch.optim as optim
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
from data_processor import DataProcessorFactory
from sklearn_processor import SKLearnProcessor
from torch_models import ModelArchitectureFactory
from artifacts_io_handler import ArtifactsIOHandler, ArtifactsMessage
from torch_dataset import TorchDataset
from eval_no_grad import EvalNoGrad
from utils import getNumWorkers, getProcessingDevice
from constants import FIXED_SEED
from sklearn_pipelines import SKLearnPipelineMaker
from safe_dict import SafeDict

logger = logging.getLogger(__name__)
    
  
class Trainer(EnforceOverrides):
  
  @abstractmethod
  def __init__(self, trainParams: SafeDict):
    self.trainParams = trainParams.copy()
    self.dataReader = DataReaderFactory.make(trainParams)
    self.dataProcessor = DataProcessorFactory.make(
      trainParams['features'], targetCol=trainParams['target']
    )
    self.artifactsIOHandler = ArtifactsIOHandler()
    
    self.inputColTypes = None
    self.valPerformanceMetrics = None
    
  @abstractmethod
  def train(self):
    pass
  
  @abstractmethod
  def saveModel(self):
    pass

  @final
  def _calcPerformanceMetrics(
    self, actuals: pd.Series, preds: np.ndarray
  ) -> T.Mapping[str, float]:
    
    if (targetType := self.trainParams['target_type']) == 'binary':
      perfCalculator = self._calcBinaryPerformanceMetrics
    
    elif targetType == 'regression':
      perfCalculator = self._calcRegressionPerformanceMetrics
    
    else:
      raise ValueError(f'{targetType=} not recognized')
      
    metrics = perfCalculator(actuals, preds)
    logger.info(f'Performance metrics:\n{metrics}')
    return metrics
      
  @final
  def _calcBinaryPerformanceMetrics(
    self, actuals: pd.Series, preds: np.ndarray
  ) -> T.Mapping[str, float]:
    metrics = {
      'confusion_matrix': confusion_matrix(actuals, preds >= 0.5),
      'roc_auc': roc_auc_score(actuals, preds),
      'pr_auc': average_precision_score(actuals, preds),
      'prop_trues': np.mean(actuals),
      'nobs': preds.shape[0]
    }
    return metrics
  
  @final
  def _calcRegressionPerformanceMetrics(
    self, actuals: pd.Series, preds: np.ndarray
  ) -> T.Mapping[str, float]:
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
  def __init__(self, trainParams: SafeDict):
    super().__init__(trainParams)
    self.sklearnProcessor = SKLearnProcessor(trainParams)
    self.modelArchitecture = (
      ModelArchitectureFactory.make(trainParams['pytorch_model'])
    )
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
    
    torchTrainDF = TorchDataset(
      trainDF, sklearnProcessor, target=self.trainParams['target']
    )
    torchValDF = TorchDataset(
      valDF, sklearnProcessor, target=self.trainParams['target']
    )
    
    batchSize = self.trainParams['batch_size']
    numWorkers = getNumWorkers(self.trainParams['num_workers'])
    logger.info(f'Running on {numWorkers} cores')
    
    trainLoader = DataLoader(
      torchTrainDF, batch_size=batchSize, num_workers=numWorkers, shuffle=False
    )
    valLoader = DataLoader(
      torchValDF, batch_size=batchSize, num_workers=numWorkers, shuffle=False
    )

    device = getProcessingDevice()
    logger.info(f'Training on {device=} with {numWorkers=}')
    
    self.model = self.modelArchitecture(sklearnProcessor.featureNames).to(device)
    self.optimizer = optim.Adam(self.model.parameters())
    lossCriterion = self._makeLossCriterion()
    
    allEpochTrainLosses = []
    allEpochValLosses = []
    
    finalEpochValPreds = torch.empty(0, dtype=torch.float)
    finalEpochValActuals = torch.empty(0, dtype=torch.float)
    
    torch.manual_seed(FIXED_SEED)
    
    for i in range(1, (numEpochs := self.trainParams['epochs'])+1):
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
      'model_type': 'pytorch',
      'model_name': self.trainParams['pytorch_model'],
      'target': self.trainParams['target'],
      'target_type': self.trainParams['target_type'],
      'val_range': self.trainParams['val_range'],
      'val_perf_metrics': self.valPerformanceMetrics,
      'input_col_types': self.inputColTypes
    }
    artifacts = {
      'sklearn_processor': self.sklearnProcessor.get(),
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
    }
    message = ArtifactsMessage(meta, artifacts)
    self.artifactsIOHandler.save(message)

  def _makeLossCriterion(self) -> None:
    
    if (targetType := self.trainParams['target_type']) == 'binary':
      lossType = nn.BCEWithLogitsLoss
      
    elif targetType == 'regression':
      lossType = nn.L1Loss
      
    else:
      raise ValueError(f'{targetType=} not recognized')
      
    return lossType(reduction='sum')
  
    
class SKLearnTrainer(Trainer):
  
  def __init__(self, trainParams: SafeDict):
    super().__init__(trainParams)
    self.sklearnPipelineMaker = SKLearnPipelineMaker(trainParams)
    
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
      f'Running hyperparameter search for {self.trainParams["n_iter"]} iterations'
    )
    pipeline.fit(trainX, trainY)
    logger.info('Training complete')
    
    valX, valY = self._splitXY(valDF)
    valPreds = pipeline.predict(valX)
    
    self.pipeline = pipeline
    self.valPerformanceMetrics = self._calcPerformanceMetrics(valY, valPreds)
    
  @overrides 
  def saveModel(self) -> None:
    meta = {
      'model_type': 'sklearn',
      'model_name': self.trainParams['sklearn_model'],
      'target': self.trainParams['target'],
      'val_range': self.trainParams['val_range'],
      'input_col_types': self.inputColTypes,
      'val_perf_metrics': self.valPerformanceMetrics
    }
    artifacts = {
      'model_pipeline': self.pipeline
    }
    message = ArtifactsMessage(meta, artifacts)
    self.artifactsIOHandler.save(message)
  
  def _splitXY(self, inDF) -> T.Tuple[pd.DataFrame, pd.Series]:
    target = self.trainParams['target']
    df = inDF.copy()
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
    
  
class TrainerFactory:
  
  @staticmethod
  def make(trainParams: SafeDict) -> T.Type[Trainer]:
    
    modelType = (
      'pytorch' if trainParams['pytorch_model'] is not None else 'sklearn'
    )

    if modelType == 'pytorch':
      Trainer_ = TorchTrainer
    
    elif modelType == 'sklearn':
      Trainer_ = SKLearnTrainer
    
    else:
      raise ValueError(f'{modelType=} not recognized')
      
    return Trainer_(trainParams)
