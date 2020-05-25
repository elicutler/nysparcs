
import typing as T
import logging
import pathlib
import pickle
import re
import torch

from collections import OrderedDict, namedtuple
from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from sagemaker.s3 import S3Uploader, S3Downloader
from botocore.exceptions import ClientError
from utils import nowTimestampStr
from constants import S3_BUCKET

logger = logging.getLogger(__name__)


# Note: creating Message outside of ArtifactsIOHandler def, instead of as static variable
# within class, because otherwise it cannot be pickled 
ArtifactsMessage = namedtuple('ArtifactsMessage', ['meta', 'artifacts'])


class ArtifactsIOHandler(EnforceOverrides):
  
  localArtifactsPath = 'artifacts/'
  s3BucketPath = f's3://{S3_BUCKET}/'
  s3ArtifactsPath = s3BucketPath + localArtifactsPath
  
  def save(self, artifactsMessage) -> None:
    artifactsPathLocal = self._saveToLocal(artifactsMessage)
    self._saveToS3(artifactsPathLocal)
    
  def load(self, artifactsName) -> ArtifactsMessage:
    artifactsPathLocal = self._loadFromS3(artifactsName)
    artifactsMessage = self._loadFromLocal(artifactsPathLocal)
    return artifactsMessage
  
  def _saveToLocal(self, artifactsMessage) -> str:
    '''
    Save artifacts locally and return path
    '''
    
    target = artifactsMessage.meta['target']
    modelType = artifactsMessage.meta['model_type']
    modelName = artifactsMessage.meta['model_name']
    parentPath = self.localArtifactsPath + f'{target}/{modelType}/{modelName}/'
    artifactsPathNoSuffix = parentPath + f'{modelName}_{nowTimestampStr()}'
    
    if not pathlib.os.path.exists(parentPath):
      pathlib.os.makedirs(parentPath)
      
    if modelType == 'pytorch':
      artifactsPath = artifactsPathNoSuffix + '.pt'
      artifactsMessage.meta['artifacts_path'] = artifactsPath
      torch.save(artifactsMessage, artifactsPath)
      
    elif modelType == 'sklearn':
      artifactsPath = artifactsPathNoSuffix + '.sk'
      artifactsMessage.meta['artifacts_path'] = artifactsPath
      with open(artifactsPath, 'wb') as file:
        pickle.dump(artifactsMessage, file, protocol=5)
        
    logger.info(f'Model artifact saved to local file: {artifactsPath}')
    return artifactsPath
  
  def _saveToS3(self, artifactsPathLocalStr) -> None:
    artifactsPathLocal = pathlib.Path(artifactsPathLocalStr)
    s3Path = self.s3BucketPath + str(artifactsPathLocal.parent)
    S3Uploader.upload(artifactsPathLocalStr, s3Path)
    logger.info(
      'Model artifact saved to s3:'
      f' {s3Path}/{artifactsPathLocal.stem}{artifactsPathLocal.suffix}'
    )
    
  def _loadFromS3(self, modelName) -> str:
    allArtifactsS3 = S3Downloader.list(self.s3ArtifactsPath)
    
    matchingArtifactsS3 = [
      a for a in allArtifactsS3
      if pathlib.Path(a).stem == modelName
    ]
    assert len(matchingArtifactsS3) > 0, f'{modelName=} not found'
    assert not len(matchingArtifactsS3) > 1, (
      f'Multiple models found for {modelName=}'
    )
    
    artifactsPathS3 = matchingArtifactsS3[0]
    artifactsPathLocal = re.sub(
      self.s3ArtifactsPath, self.localArtifactsPath, artifactsPathS3
    )
    artifactsParentPathLocal = pathlib.Path(artifactsPathLocal).parent
    S3Downloader.download(artifactsPathS3, artifactsParentPathLocal)
    
    logger.info(
      f'Downloaded artifact from {artifactsPathS3} to {artifactsPathLocal}'
    )
    return artifactsPathLocal
  
  def _loadFromLocal(self, artifactsPathLocalStr) -> ArtifactsMessage:
    artifactsPath = pathlib.Path(artifactsPathLocalStr)
    
    if artifactsPath.suffix == '.pt':
      return torch.load(artifactsPath)
    
    elif artifactsPath.suffix == '.sk':
      with open(artifactsPath, 'rb') as artifacts:
        return pickle.loads(artifacts.read())
      
    else:
      raise ValueError(f'file extension {artifactsPath.suffix} not recognized')
        
    

    
#   def _localSaveTorch(
#     self, artifacts, returnModelPath=False
#   ) -> T.Union[None, pathlib.Path]:
    
#     artifactsDir = pathlib.Path('artifacts/pytorch/')
#     modelName = self.params['pytorch_model']
#     thisModelDir = artifactsDir/modelName
    
#     if not pathlib.os.path.exists(thisModelDir):
#       pathlib.os.makedirs(thisModelDir, exist_ok=True)
    
#     modelPath = thisModelDir/f'{modelName}_{nowTimestampStr()}.pt'

#     torch.save(artifacts, modelPath)
#     logger.info(f'Saving model artifacts to local path: {modelPath}')
    
#     if returnModelPath:
#       return modelPath
    
#   @final
#   def _localSaveSKLearn(
#     self, artifacts, returnModelPath=False
#   ) -> T.Union[None, pathlib.Path]:

#     artifactsDir = pathlib.Path('artifacts/sklearn')
#     modelName = self.params['sklearn_model']
#     thisModelDir = artifactsDir/modelName
    
#     if not pathlib.os.path.exists(thisModelDir):
#       pathlib.os.makedirs(thisModelDir, exist_ok=True)
      
#     modelPath = thisModelDir/f'{modelName}_{nowTimestampStr()}.sk'
#     with open(modelPath, 'wb') as file:
#       pickle.dump(artifacts, file, protocol=5)
      
#     logger.info(f'Saving model artifacts to local path: {modelPath}')

#     if returnModelPath:
#       return modelPath
    
#   @final
#   def _localLoadTorch(self, modelFile=None) -> OrderedDict:
    
#     artifactsDir = pathlib.Path('artifacts/pytorch/')
#     modelName = self.params['pytorch_model']
    
#     if self.params['load_latest_state_dict'] and modelFile is None:
      
#       if (
#         modelName not in (artifactsDirContents := pathlib.os.listdir(artifactsDir))
#       ):
#         logger.warning(f'No previous artifacts found for {modelName=}')
#         return
      
#       else:
#         artifacts = [
#           a for a in pathlib.os.listdir(artifactsDir/modelName)
#           if a.startswith(modelName)
#         ]
        
#         if len(artifacts) == 0:
#           logger.warning(f'No previous artifacts found for {modelName=}')
#           return
#         else:
#           artifacts.sort(reverse=True)
        
#     else:
#       targetModel = self.params['load_state_dict'] if modelFile is None else modelFile
#       artifacts = [
#         a for a in pathlib.os.listdir(artifactsDir/modelName)
#         if (
#           re.sub('\.pt|\.pth', '', targetModel) 
#           == re.sub('\.pt|\.pth', '', a)
#         )
#       ]
#       assert len(artifacts) > 0, f'{targetModel=} not found'
#       assert len(artifacts) < 2, f'multiple artifacts found for {targetModel=}'
      
#     artifactsPath = artifactsDir/modelName/artifacts[0]
#     logger.info(f'Loading model artifacts from local path: {artifactsPath}')
#     return torch.load(artifactsPath)

# class LocalArtifactsIOHandler(ArtifactsIOHandler):
  
#   @overrides
#   def __init__(self, params):
#     super().__init__(params)
    
# #   @overrides
#   def saveTorch(self, artifacts) -> None:
#     self._localSaveTorch(artifacts)
    
# #   @overrides
#   def loadTorch(self) -> OrderedDict:
#     return self._localLoadTorch()
    
# #   @overrides
#   def saveSKLearn(self, artifacts) -> None:
#     self._localSaveSKLearn(artifacts)    


# class S3ArtifactsIOHandler(ArtifactsIOHandler):
  
# #   @overrides
#   def __init__(self, params):
#     self.params = params.copy()
#     self.s3ProjectRootPath = f's3://{S3_BUCKET}/nysparcs/'
    
# #   @overrides
#   def saveTorch(self, artifacts) -> None:
#     modelPath = self._localSaveTorch(artifacts, returnModelPath=True)
    
#     s3Path = self.s3ProjectRootPath + str(modelPath.parent)
#     S3Uploader.upload(str(modelPath), s3Path)
#     logger.info(f'Uploading model to s3: {s3Path}/{modelPath.stem}{modelPath.suffix}')
    
# #   @overrides
#   def loadTorch(self) -> OrderedDict:
    
#     # Note: cannot use pathlib.Path here because need to preserve consecutive 
#     # slashes in 's3://' 
#     artifactsDir = self.s3ProjectRootPath + 'artifacts/pytorch/'
#     modelName = self.params['pytorch_model']
#     artifactsModelDir = artifactsDir + modelName
    
#     if self.params['load_latest_state_dict']:
      
#       artifactsDirContents = S3Downloader.list(artifactsDir)
#       modelName = self.params['pytorch_model']
      
#       for a in artifactsDirContents:
#         if len(re.findall(f'/{modelName}_', a)) > 0:
#           break
#       else:
#         logger.warning(f'No previous artifacts found for {modelName=}')
#         return
      
#       artifacts = [
#         a for a in artifactsDirContents
#         if  len(re.findall(f'/{modelName}_', a)) > 0
#       ]

#       if len(artifacts) == 0:
#         logger.warning(f'No previous artifacts found for {modelName=}')
#         return
#       else:
#         artifacts.sort(reverse=True)
        
#     elif (targetModel := self.params['load_state_dict']) is not None:
#       artifacts = [
#         a for a in S3Downloader.list(artifactsModelDir)
#         if (
#           re.sub('\.pt|\.pth', '', targetModel) 
#           == re.sub('\.pt|\.pth', '', a)
#         )
#       ]
#       assert len(artifacts) > 0, f'{targetModel=} not found'
#       assert len(artifacts) < 2, f'multiple artifacts found for {targetModel=}'
      
#     else:
#       raise Exception(
#         'Invalid combination of load_latest_state_dict and load_state_dict args'
#       )
      
#     artifactsPath = artifacts[0]
#     logger.info(f'Loading model artifacts locally from s3 path: {artifactsPath}')
    
#     localDir = f'artifacts/pytorch/{modelName}/'
#     S3Downloader.download(artifactsPath, localDir)
    
#     modelFile = pathlib.Path(artifactsPath).stem + pathlib.Path(artifactsPath).suffix
#     return self._localLoadTorch(modelFile)
    
# #   @overrides
#   def saveSKLearn(self, artifacts) -> None:
#     modelPath = self._localSaveSKLearn(artifacts, returnModelPath=True)
    
#     s3Path = self.s3ProjectRootPath + str(modelPath.parent)
#     S3Uploader.upload(str(modelPath), s3Path)
#     logger.info(f'Uploading model to s3: {s3Path}/{modelPath.stem}{modelPath.suffix}')

#   def getAllModelArtifacts(self) -> T.List[str]:
    
#     def listAllArtifactPaths(path, list_) -> None:
#       for item in S3Downloader.list(path):
#         if re.findall('\.pt$|\.sk$', item):
#           list_.append(item)
#         else:
#           listAllArtifactPaths(item)
    
#     rootPath = self.s3ProjectRootPath + 'artifacts'
#     modelArtifacts = []
#     listAllArtifactPaths(rootPath, modelArtifacts)
    
#     return modelArtifacts
  
#   def downloadModelsFromList(self, modelsList) -> None:
#     for m in modelsList:
#       if m.endswith('.pt'):
#         localDir = (
#           pathlib.Path('artifacts/pytorch')/pathlib.Path(m).parent.stem
#         )
#       elif m.endswith('.sk'):
#         localDir = (
#           pathlib.Path('artifacts/sklearn')/pathlib.Path(m).parent.stem
#         )
#       S3Downloader.download(m, localDir)
    
# class ArtifactsIOHandlerFactory:
  
#   @staticmethod
#   def make(params) -> T.Type[ArtifactsIOHandler]:
    
#     if (artifactsEnv := params['artifacts_env']) == 'local':
#       artifactsIOHandler = LocalArtifactsIOHandler
      
#     elif artifactsEnv == 's3':
#       artifactsIOHandler = S3ArtifactsIOHandler
      
#     else:
#       raise ValueError(f'{artifactsEnv} not recognized')
      
#     return artifactsIOHandler(params)