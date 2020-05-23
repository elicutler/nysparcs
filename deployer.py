
import typing as T
import logging 
import argparse
import pathlib
import re
import torch
import pickle

from artifacts_io_handler import S3ArtifactsIOHandler

logger = logging.getLogger(__name__)


class Deployer:
  
  def __init__(self, params):
    self.params = params.copy()
    self.artifactsIOHandler = S3ArtifactsIOHandler(params)
    
  def deploy(self) -> None:
    
    allS3Models = self.artifactsIOHandler.getAllModelArtifacts()
    self.artifactsIOHandler.downloadModelsFromList(allS3Models)
    allLocalModels = self._getAllLocalModels()
    inRangeModels = (
      allLocalModels if (modelType := self.params['model_type']) is None
      else self._keepModelsOfType(allLocalModels, 'pytorch')
      if modelType == 'pytorch'
      else self._keepModelsOfType(allLocalModels, 'sklearn')
      if modelType == 'sklearn'
      else None
    )
    assert inRangeModels is not None
    
    fullModelPath = (
      self._selectModelNameFromList(inRangeModels)
      if self.params['model_name'] is not None
      else self._selectBestModelNameFromList(inRangeModels)
    )
    breakpoint()
    
#     modelName, modelArtifacts = (
#       self.params['model_name'] 
#       or self._getBestModel(inRangeModels)
#     )
#     model = self._loadBestModel(modelType, modelConfig)
    
#     model.deploy(1, 'ml.t2.medium', endpoint_name=model)

  def _getAllLocalModels(self) -> T.List[str]:
    
    rootPath = pathlib.Path('artifacts/')
    
    def appendFilesRecursively(path, list_) -> None:
      
      for item in pathlib.os.listdir(path):
        if re.findall('\.pt$|\.sk$', item):
          list_.append(path/item)
        elif pathlib.os.path.isdir(path/item):
          appendFilesRecursively(path/item, list_)
    
    models = []
    appendFilesRecursively(rootPath, models)
    return models
  
  def _selectModelNameFromList(self, inRangeModels) -> str:
    matchingModelPaths = [
      m for m in inRangeModels
      if pathlib.Path(m).stem == self.params['model_name']
    ]
    assert len(matchingModelPaths) > 0, (
      f'Model {self.params["model_name"]} not found among models:'
      f' {inRangeModels}'
    )
    assert len(matchingModelPaths) < 2, (
      f'Multiple models found matching name {self.params["model_name"]}:'
      f'{matchingModelPaths}'
    )
    return matchingModelPaths[0]
  
  def _keepModelsOfType(self, modelsList, modelType) -> T.List[str]:
    if modelType == 'pytorch':
      return [m for m in modelsList if m.endswith('.pt')]
    elif modelType == 'sklearn':
      return [m for m in modelsList if m.endswith('.sk')]
    else:
      raise ValueError(f'{modelType=} not recognized')
      
  def _selectBestModelNameFromList(self, modelsList) -> T.Type[pathlib.Path]:
    
    modelsAndPerf = []
    for m in modelsList:
      if m.suffix == '.pt':
        modelsAndPerf.append((m, torch.load(m)))
      elif m.suffix == '.sk':
        with open(m.as_posix(), 'rb') as file:
          modelBytes = file.read()
        modelsAndPerf.append((m, pickle.loads(modelBytes)))
          
    bestModel = None
    bestScore = 0
    for m in modelsAndPerf:
      if (score := m[1]['val_perf_metrics']['pr_auc']) > bestScore:
        bestModel = m[0]
        bestScore = score
      
    assert bestModel is not None
    return bestModel
      
    
    