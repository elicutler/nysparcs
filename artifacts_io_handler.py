
import typing as T
import logging
import pathlib
import pickle
import re
import torch

from collections import OrderedDict
from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from utils import nowTimestampStr

logger = logging.getLogger(__name__)


class ArtifactsIOHandler(EnforceOverrides):
  
  @abstractmethod
  def __init__(self, params):
    self.params = params.copy()
    
  @abstractmethod
  def saveTorch(self, artifacts) -> None:
    raise NotImplementedError
    
  @abstractmethod
  def loadTorch(self) -> OrderedDict:
    raise NotImplementedError
    
  @abstractmethod
  def saveSKLearn(self, artifacts) -> None:
    raise NotImplementedError


class LocalArtifactsIOHandler(ArtifactsIOHandler):
  
  @overrides
  def __init__(self, params):
    super().__init__(params)
    
  @overrides
  def saveTorch(self, artifacts) -> None:
    
    artifactsDir = pathlib.Path('artifacts/pytorch/')
    modelName = self.params['pytorch_model']
    thisModelDir = artifactsDir/modelName
    
    if not pathlib.os.path.exists(thisModelDir):
      pathlib.os.makedirs(thisModelDir, exist_ok=True)
    
    modelPath = thisModelDir/f'{modelName}_{nowTimestampStr()}.pt'

    torch.save(artifacts, modelPath)
    logger.info(f'Saving model artifacts to {modelPath}')
    
  @overrides
  def loadTorch(self) -> OrderedDict:
  
    artifactsDir = pathlib.Path('artifacts/pytorch/')
    modelName = self.params['pytorch_model']
    
    if self.params['load_latest_state_dict']:
      
      if (
        (modelName := self.params['pytorch_model'])
        not in (artifactsDirContents := pathlib.os.listdir(artifactsDir))
      ):
        logger.warning(f'No previous state dicts found for {modelName=}')
        return
      
      else:
        artifacts = [
          a for a in pathlib.os.listdir(artifactsDir/modelName)
          if a.startswith(modelName)
        ]
        
        if len(artifacts) == 0:
          logger.warning(f'No previous state dicts found for {modelName=}')
          return
        else:
          artifacts.sort(reverse=True)
        
    elif (targetModel := self.params['load_state_dict']) is not None:
      artifacts = [
        a for a in pathlib.os.listdir(artifactsDir/modelName)
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
      
    artifactsPath = artifactsDir/modelName/artifacts[0]
    logger.info(f'Loading model and optimizer state dicts from {artifactsPath}')
    return torch.load(artifactsPath)
    
  @overrides
  def saveSKLearn(self, artifacts) -> None:

    artifactsDir = pathlib.Path('artifacts/sklearn')
    modelName = self.params['sklearn_model']
    thisModelDir = artifactsDir/modelName
    
    if not pathlib.os.path.exists(thisModelDir):
      pathlib.os.makedirs(thisModelDir, exist_ok=True)
      
    modelPath = thisModelDir/f'{modelName}_{nowTimestampStr()}.sk'
    with open(modelPath, 'wb') as file:
      pickle.dump(artifacts, file, protocol=5)
      
    logger.info(f'Saving model artifacts to {modelPath}')
    


class S3ArtifactsIOHandler(ArtifactsIOHandler):
  
  @overrides
  def __init__(self, params):
    self.params = params.copy()
    
  @overrides
  def saveTorch(self, artifacts) -> None:
    
#     artifactsDir = pathlib.Path('artifacts/pytorch/')
#     modelName = self.params['pytorch_model']
#     thisModelDir = artifactsDir/modelName
    
#     if not pathlib.os.path.exists(thisModelDir):
#       pathlib.os.makedirs(thisModelDir, exist_ok=True)
    
#     modelPath = thisModelDir/f'{modelName}_{nowTimestampStr()}.pt'

#     torch.save(artifacts, modelPath)
#     logger.info(f'Saving model artifacts to {modelPath}')
    
  @overrides
  def loadTorch(self) -> OrderedDict:
    pass
    
  @overrides
  def saveSKLearn(self, artifacts) -> None:
    pass
    

class ArtifactsIOHandlerFactory:
  
  @staticmethod
  def make(params) -> T.Type[ArtifactsIOHandler]:
    
    if (artifactsEnv := params['artifacts_env']) == 'local':
      artifactsIOHandler = LocalArtifactsIOHandler
      
    else:
      raise ValueError(f'{artifactsEnv} not recognized')
      
    return artifactsIOHandler(params)