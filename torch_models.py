
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
  
  def __init__(self, featureNames) -> None:
    super().__init__()
    self.featureNames = featureNames
    self.catFeatureIndexRangeMap = self._makeCatFeatureIndexRangeMap()
    self.numFeatureIndexRange = (0, self._getMinCatFeatureIndex()) 
    
    self.catEmbeddingLayers = self._makeCatEmbeddingLayers()
  
  def forward(self, X) -> torch.Tensor:
    
    numLayer = X[:, self.numFeatureIndexRange[0]:self.numFeatureIndexRange[1]]
    catSubsets = []
    for (_, (idxMin, idxMax)), layer in self.catEmbeddingLayers.items():
      catSubsets.append(layer(X[:, idxMin:idxMax+1]))
      
    catEmbeddingLayersFused = torch.cat(catSubsets, dim=1)
    X = torch.cat([numLayer, catEmbeddingLayersFused], dim=1)
    
    breakpoint()
    
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
  
  def _makeCatEmbeddingLayers(self):
    
    catEmbeddingLayers = {}
    for catFeature, (idxMin, idxMax) in self.catFeatureIndexRangeMap.items():
      
      inNodes = idxMax - idxMin + 1
      outNodes = round(inNodes**(1/2))
      linearLayer = nn.Linear(inNodes, outNodes)
      catEmbeddingLayers[(catFeature, (idxMin, idxMax))] = linearLayer
      
    return catEmbeddingLayers

  
    
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()