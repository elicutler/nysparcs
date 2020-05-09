
import typing as T
import logging

from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from target_encoder import TargetEncoder

logger = logging.getLogger(__name__)


class SKLearnPipelineMaker:
  
  def __init__(self, params):
    self.params = params.copy()
    self.catEncoder = self._getCatEncoder()
    
    self.inputColTypes = None
    self.catFeatureCols = None
    self.numFeatureCols = None
    
  def loadInputColTypes(self, inputColTypes) -> None:
    self.inputColTypes = inputColTypes.copy()
    self.catFeatureCols = (
      self.inputColTypes[self.inputColTypes == 'object'].index.to_list()
    )
    self.numFeatureCols = [
      c for c in self.inputColTypes.index if c not in self.catFeatureCols
    ]
    
  def makePipeline(self) -> Pipeline:
    
    if (pipelineType := self.params['sklearn_pipeline']) == 'xgboost':
      return self._makeXGBoostPipeline()
    
    else:
      raise ValueError(f'{pipelineType=} not recognized')

  def _getCatEncoder(self) -> T.Union[TargetEncoder, OneHotEncoder]:

    if (catEncoderStrat := self.params['cat_encoder_strat']) == 'target':
      return TargetEncoder(priorFrac=0.1)

    elif catEncoderStrat == 'one_hot':
      return OneHotEncoder(handle_unknown='ignore')

    else:
      raise ValueError(f'{catEncoderStrat=} not recognized')
      
  def _makeXGBoostPipeline(self) -> Pipeline:
    pass
    
  
#     numFeatureCols = trainX.select_dtypes(include=['number']).columns
#     catFeatureCols = trainX.select_dtypes(include=['object']).columns
    
#     catEncoder = self._getCatEncoderStrat()
    
#     numPipe = Pipeline([
#       ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
#       ('scaler', StandardScaler())
#     ])
#     catPipe = Pipeline([
#       ('imputer', SimpleImputer(strategy='constant', fill_value='__UNK__')),
#       ('cat_encoder', catEncoder)
#     ])
#     pipe = ColumnTransformer([
#       ('num_pipe', numPipe, numFeatureCols),
#       ('cat_pipe', catPipe, catFeatureCols)
#     ])
#     pipe.fit(trainX, trainY)
    
#     self.pipe = pipe
#     self._setFeatureNames(trainX, numFeatureCols, catFeatureCols)