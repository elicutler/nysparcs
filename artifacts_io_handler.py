
import logging

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)


class ArtifactsIOHandler(EnforceOverrides):
  
  def __init__(self, params):
    self.params = params.copy()


class LocalArtifactsIOHandler(ArtifactsIOHandler):
  pass


class AWSArtifactsIOHandler(ArtifactsIOHandler):
  pass


class ArtifactsIOHandlerFactory:
  pass