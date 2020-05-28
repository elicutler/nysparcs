
import typing as T
import logging

from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from target_encoder import TargetEncoder
from utils import getNumWorkers
from constants import FIXED_SEED

logger = logging.getLogger(__name__)


class SKLearnPipelineMaker:
  
  def __init__(self, params):
    self.params = params.copy()
    self.catEncoder = self._getCatEncoder()
    
    self.target = params['target']
    self.n_iter = params['n_iter']
    self.n_jobs = getNumWorkers(self.params['num_workers'])
    self.eval_metric = params['eval_metric']
    
    self.inputColTypes = None
    self.catFeatureCols = None
    self.numFeatureCols = None
    
  def loadInputColTypes(self, inputColTypes) -> None:
    self.inputColTypes = inputColTypes.copy()
    self.catFeatureCols = (
      self.inputColTypes.drop(self.target)[self.inputColTypes == 'object']
      .index.to_list()
    )
    self.numFeatureCols = [
      c for c in self.inputColTypes.drop(self.target).index 
      if c not in self.catFeatureCols
    ]
    
  def makePipeline(self) -> Pipeline:
    
    numPipe = Pipeline([
      ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
      ('scaler', StandardScaler())
    ])
    catPipe = Pipeline([
      ('imputer', SimpleImputer(strategy='constant', fill_value='__UNK__')),
      ('cat_encoder', self.catEncoder)
    ])
    processorPipe = ColumnTransformer([
      ('num_pipe', numPipe, self.numFeatureCols),
      ('cat_pipe', catPipe, self.catFeatureCols)
    ])
    
    if (modelType := self.params['sklearn_model']) == 'gradient_boosting_classifier':
      return self._makeGBCEstimator(processorPipe)
    
    else:
      raise ValueError(f'{modelType=} not recognized')

  def _getCatEncoder(self) -> T.Union[TargetEncoder, OneHotEncoder]:

    if (catEncoderStrat := self.params['cat_encoder_strat']) == 'target':
      return TargetEncoder(priorFrac=0.1)

    elif catEncoderStrat == 'one_hot':
      return OneHotEncoder(handle_unknown='ignore')

    else:
      raise ValueError(f'{catEncoderStrat=} not recognized')
      
  def _makeGBCEstimator(self, processorPipe) -> RandomizedSearchCV: 
    
    estimatorPipe = Pipeline([
      ('processor', processorPipe),
      ('gbc', GradientBoostingClassifier())
    ])
    paramDistributions = {
      'processor__cat_pipe__cat_encoder__priorFrac': [0., 0.01, 0.1, 0.5, 1.],
      'gbc__learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.],
      'gbc__max_depth': [2, 3, 4, 6],
      'gbc__max_features': [0.25, 0.5, 0.75, 1.],
      'gbc__min_samples_leaf': [1, 2, 4],
      'gbc__min_samples_split': [2, 4, 8],
      'gbc__n_estimators': [100, 500, 1000],
      'gbc__subsample': [0.1, 0.5, 0.8, 1.]
    }
    hyperparamSearchPipe = RandomizedSearchCV(
      estimatorPipe, paramDistributions, n_iter=self.n_iter, 
      scoring=self.eval_metric, n_jobs=self.n_jobs, random_state=FIXED_SEED,
      verbose=1
    )
    logger.info(f'Pipeline created with {self.n_jobs=}')
    return hyperparamSearchPipe