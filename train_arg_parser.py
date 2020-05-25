
import typing as T
import logging
import argparse
import json

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from constants import TARGET_OPTIONS

logger = logging.getLogger(__name__)


class TrainArgParser:
  
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
    self.parser.add_argument('--target', type=str)
    self.parser.add_argument(
      '--target_type', type=str, help=(
        'Type of target variable. Options: \'binary\', \'regression\''
      )
    )
    self.parser.add_argument(
      '--features', type=str, nargs='+', default=None, help=(
        'If None, use all cleaned column names in dataset except for target.'
        ' (Default: None)'
      )
    )
    self.parser.add_argument(
      '--cat_encoder_strat', type=str, help=(
        'Encoding for categorical features. Options: \'one_hot\', \'target\''
      )
    )
    self.parser.add_argument('--pytorch_model', type=str)
    self.parser.add_argument('--sklearn_model', type=str)
    self.parser.add_argument('--local_data_path', type=str)
    self.parser.add_argument('--socrata_data_key', type=str)
    self.parser.add_argument('--s3_data_path', type=str)
    self.parser.add_argument(
      '--eval_metric', type=str, help=(
        'Scoring metric for sk-learn model selection'
      )
    )
    self.parser.add_argument(
      '--train_range', type=int, nargs=2, default=[0, 999],
      help='Index range of training records'
    )
    self.parser.add_argument(
      '--val_range', type=int, nargs=2, default=[1000, 1999],
      help='Index range of hold-out validation records'
    )
    self.parser.add_argument(
      '--epochs', type=int, help='Number of epochs to train pytorch model for.'
    )
    self.parser.add_argument(
      '--batch_size', type=int, help='Batch size for pytorch models.'
    )
    self.parser.add_argument(
      '--n_iter', type=int, help=(
        'Number of iterations for randomized search of sklearn models.'
      )
    )
    self.parser.add_argument(
      '--num_workers', type=int, default=-1, 
      help='-1: Use all but one core for training. (Default: -1)'
    )
    self.parser.add_argument(
      '--run_id', type=str, 
      help=(
        'Read parser arguments from stored json in run_config_store.json'
        ' with run_id as key. If this argument is passed, all other'
        ' arguments are ignored.'
      )
    )
    
  def parseArgs(self, argsInput=None) -> argparse.Namespace:
    
    rawArgs = self.parser.parse_args(args=argsInput)
    
    if rawArgs.run_id is None:
      args = rawArgs
    else:
      args = self._parseArgsFromRunID(rawArgs.run_id)
      
    self._validateArgs(args)
    return args
  
  def _parseArgsFromRunID(self, runID) -> argparse.Namespace:
    
    with open('run_config_store.json') as file:
      runConfigs = json.load(file)
      
    argsList = self._argsDictToList(runConfigs, runID)
    args = self.parser.parse_args(argsList)
    
    self._validateArgs(args)
    return args
  
  def _argsDictToList(self, runConfigs, runID) -> T.List[str]:
    '''
    Convert {'key': <values any type> <, ...>} 
    to ['--key' <, 'v1'> <, ..., 'vn'> <, ...>]
    '''
    argsDict = {**runConfigs[runID], **{'run_id': runID}}
    listOfTuples = [
      (f'--{k}', str(v)) if not isinstance(v, (bool, list))
      else (f'--{k}',) + tuple(str(e) for e in v) if isinstance(v, list)
      else (f'--{k}',) 
      for k, v in argsDict.items() 
      if not (isinstance(v, bool) and v is False)
    ]
    argsList = [e for t in listOfTuples for e in t]
    return argsList

  def _validateArgs(self, args) -> None:
        
    assert args.target in TARGET_OPTIONS
    # categorical feature encoding
    assert args.cat_encoder_strat in ['one_hot', 'target']
    # pytorch model XOR sklearn model
    assert bool(args.pytorch_model) + bool(args.sklearn_model) == 1
    # local data path XOR socrata data key XOR s3_data_path
    assert (
      bool(args.local_data_path) 
      + bool(args.socrata_data_key)
      + bool(args.s3_data_path)
    ) == 1
    # non-overlapping train/test intervals with start <= end  
    assert args.train_range[0] <= args.train_range[1]
    assert args.val_range[0] <= args.val_range[1]
    assert (
      (args.train_range[1] <= args.val_range[0])
      or (args.val_range[1] <= args.train_range[0])
    )
    # only set epochs and/or batch_size for pytorch models
    assert bool(args.pytorch_model) is bool(args.epochs) is bool(args.batch_size)
    # only pass eval_metric and n_iter for sklearn models
    assert bool(args.sklearn_model) is bool(args.eval_metric) is bool(args.n_iter)
