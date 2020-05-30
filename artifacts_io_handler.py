
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
from utils import initializeSession, parseSecrets, nowTimestampStr
from constants import LOCAL_ARTIFACTS_PATH

logger = logging.getLogger(__name__)


# Note: creating Message outside of ArtifactsIOHandler def, instead of as static variable
# within class, because otherwise it cannot be pickled 
ArtifactsMessage = namedtuple('ArtifactsMessage', ['meta', 'artifacts'])


class ArtifactsIOHandler:
  
  def __init__(self):
    self.session = initializeSession()
    self.s3BucketPath = f"s3://{parseSecrets()['s3_bucket']}"
    self.s3ArtifactsPath = self.s3BucketPath + LOCAL_ARTIFACTS_PATH
  
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
    parentPath = LOCAL_ARTIFACTS_PATH + f'{target}/{modelType}/{modelName}/'
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
    S3Uploader.upload(artifactsPathLocalStr, s3Path, session=self.session)
    logger.info(
      'Model artifact saved to s3:'
      f' {s3Path}/{artifactsPathLocal.stem}{artifactsPathLocal.suffix}'
    )
    
  def _loadFromS3(self, modelName) -> str:
    allArtifactsS3 = (
      S3Downloader.list(self.s3ArtifactsPath, session=self.session)
    )
    
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
      self.s3ArtifactsPath, LOCAL_ARTIFACTS_PATH, artifactsPathS3
    )
    artifactsParentPathLocal = pathlib.Path(artifactsPathLocal).parent
    S3Downloader.download(
      artifactsPathS3, artifactsParentPathLocal, session=self.session
    )
    
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
        
    
