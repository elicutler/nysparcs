
import typing as T
import logging 
import argparse

from artifacts_io_handler import S3ArtifactsIOHandler

logger = logging.getLogger(__name__)


class Deployer:
  
  def __init__(self, params):
    self.params = params.copy()
    self.artifactsIOHandler = S3ArtifactsIOHandler(params)
    
  def deploy(self) -> None:
    
    inRangeModels = (
      (allModels := self._getAllModels()) 
      if (modelType := self.params['model_type']) is None
      else self._keepModelsOfType(allModels, 'pytorch')
      if modelType == 'pytorch'
      else self._keepModelsOfType(allModels, 'sklearn')
      if modelType == 'sklearn'
      else None
    )
    assert inRangeModels is not None
    
    breakpoint()
    fullModelPath = (
      self._selectModelNameFromList(inRangeModels)
      if self.params['model_name'] is not None
      else self._selectBestModelNameFromList(inRangeModels)
    )
    
#     modelName, modelArtifacts = (
#       self.params['model_name'] 
#       or self._getBestModel(inRangeModels)
#     )
#     model = self._loadBestModel(modelType, modelConfig)
    
#     model.deploy(1, 'ml.t2.medium', endpoint_name=model)
    
  def _getAllModels(self) -> T.List[str]:
    allModels = self.artifactsIOHandler.getAllModelArtifacts()
    return allModels
  
  def _selectModelNameFromList(inRangeModels) -> str:
    matchingModelPaths = [
      m for m in inRangeModels
      if pathlib.Path(m).stem == self.params['model_name']
    ]
    assert
    
    