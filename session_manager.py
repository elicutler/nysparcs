
import typing as T
import logging
import boto3
import sagemaker

from utils import parseSecrets

logger = logging.getLogger(__name__)


class SessionManager:
  
  def __init__(self):
    secrets = parseSecrets()
    boto3Session = boto3.Session(
      aws_access_key_id=secrets['aws_access_key_id'],
      aws_secret_access_key=secrets['aws_secret_access_key'],
      region_name=secrets['region_name']
    )
    self.sagemakerSession = sagemaker.session.Session(
      boto_session=boto3Session
    )
    
  def getSagemakerSession(self) -> sagemaker.session.Session:
    return self.sagemakerSession
    
    
    
