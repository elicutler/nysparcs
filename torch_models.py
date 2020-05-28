
import typing as T
import logging
import re
import torch.nn as nn
import torch.nn.functional as F
import torch
import utils

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)
    

class CatEmbedNet(nn.Module):
  
  def __init__(self, featureNames):
    super().__init__()
    self.featureNames = featureNames
    self.catFeatureIndexRangeMap = self._makeCatFeatureIndexRangeMap()
    self.numFeatureIndexRange = (0, self._getMinCatFeatureIndex()) 
    
    self.catEmbeddingLayers = self._makeCatEmbeddingLayers()
    self.catEmbedActivation = nn.SELU()
    Layer0NodesOut = self._catEmbeddingPlusNumNodes()
    self.fullyConnectedLayer1 = nn.Linear(Layer0NodesOut, Layer0NodesOut)
    self.activationLayer1 = nn.SELU()
    self.dropoutLayer1 = nn.AlphaDropout(p=0.5)
    self.fullyConnectedLayer2 = nn.Linear(Layer0NodesOut, 1)
    self.activationLayer2 = None
  
  def forward(self, X) -> torch.Tensor:
    
    numLayer = X[:, self.numFeatureIndexRange[0]:self.numFeatureIndexRange[1]]
    catSubsets = []
    for (_, (idxMin, idxMax)), layer in self.catEmbeddingLayers.items():
      XSubsetLayer = layer(X[:, idxMin:idxMax+1])
      XSubsetLayer = self.catEmbedActivation(XSubsetLayer)
      catSubsets.append(XSubsetLayer)
      
    catEmbeddingLayersFused = torch.cat(catSubsets, dim=1)
    X = torch.cat([numLayer, catEmbeddingLayersFused], dim=1)
    X = self.fullyConnectedLayer1(X)
    X = self.activationLayer1(X)
    X = self.dropoutLayer1(X)
    X = self.fullyConnectedLayer2(X)
    
    return X
    
  def _makeCatFeatureIndexRangeMap(self) -> T.Dict[str, T.Tuple[int]]:
    catFeatures = sorted({
      c.group() for f in self.featureNames 
      if (c := re.search('^x\d+_', f))
    })
    catFeatureIndexRangeMap = {
      c: utils.getMinMaxIndicesOfItemInList(c, self.featureNames, startsWith=True)
      for c in catFeatures
    }
    return catFeatureIndexRangeMap
    
  def _getMinCatFeatureIndex(self) -> int:
    allIndices = utils.flattenNestedSeq(self.catFeatureIndexRangeMap.values())
    return min(allIndices)
  
  def _makeCatEmbeddingLayers(self) -> dict:
    
    catEmbeddingLayers = {}
    for catFeature, (idxMin, idxMax) in self.catFeatureIndexRangeMap.items():
      
      inNodes = idxMax - idxMin + 1
      outNodes = round(inNodes**(1/2))
      linearLayer = nn.Linear(inNodes, outNodes)
      catEmbeddingLayers[(catFeature, (idxMin, idxMax))] = linearLayer
      
    return catEmbeddingLayers
  
  def _catEmbeddingPlusNumNodes(self) -> int:
    numFeatures = self.numFeatureIndexRange[1] - self.numFeatureIndexRange[0]
    catFeatures = sum(
      [layer.out_features for layer in self.catEmbeddingLayers.values()]
    )
    return numFeatures + catFeatures
  
  
class ModelArchitectureFactory:
  
  @staticmethod
  def make(architecture) -> T.Type[nn.Module]:
  
    if architecture == 'CatEmbedNet':
      return CatEmbedNet

    else:
      raise ValueError(f'{architecture=} not recognized')

