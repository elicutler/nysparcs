
import typing as T
import logging
import argparse
import json

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final

logger = logging.getLogger(__name__)


class TrainArgParser:
  
  def __init__(self) -> None:
    self.parser = argparse.ArgumentParser()
    
    self.parser.add_argument('--target', type=str, default='prior_auth_dispo')
    self.parser.add_argument(
      '--features', type=str, nargs='+', default=None, help=(
        'If None, use all cleaned column names in dataset except for target.'
        ' (Default: None)'
      )
    )
    self.parser.add_argument(
      '--cat_encoder_strat', type=str, default='one_hot', help=(
        'Encoding for categorical features.'
        ' Options: \'one_hot\', \'target\' (Default: \'one_hot\')'
      )
    )
#     self.parser.add_argument(
#       '--target_encoder_prior', type=float, default=0., help=(
#         'Regularization parameter for categorical feature target'
#         ' encoding to pull level means toward grand mean. Only set'
#         ' this if cat_encoder_strat=\'target\'. (Default: 0.)'
#       )
#     )
    self.parser.add_argument('--pytorch_model', type=str)
    self.parser.add_argument('--sklearn_model', type=str)
    self.parser.add_argument('--local_data_path', type=str)
    self.parser.add_argument('--socrata_data_key', type=str)
    self.parser.add_argument(
      '--load_latest_state_dict', action='store_true', help=(
        'Load latest model and optimizer state_dicts for pytorch model'
        ' of specified type. If none is found, do nothing.'
      )
    )
    self.parser.add_argument(
      '--load_state_dict', type=str, help=(
        'Load specified state_dict of model and optimizer for'
        ' pytorch model of specified type. If not found, an error is thrown.'
      )
    )
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
    self.parser.add_argument('--deploy', action='store_true')
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
        
    assert args.target in ['prior_auth_dispo', 'length_of_stay']
    # categorical feature encoding
    assert args.cat_encoder_strat in ['one_hot', 'target']
#     assert not (
#       args.cat_encoder_strat == 'one_hot' and args.target_encoder_prior > 0.
#     )
#     assert args.target_encoder_prior >= 0.
    # pytorch model xor sklearn model
    assert bool(args.pytorch_model) + bool(args.sklearn_model) == 1
    # local data path xor internet data key
    assert bool(args.local_data_path) + bool(args.socrata_data_key) == 1
    # non-overlapping train/test intervals with start <= end  
    assert args.train_range[0] <= args.train_range[1]
    assert args.val_range[0] <= args.val_range[1]
    assert (
      (args.train_range[1] <= args.val_range[0])
      or (args.val_range[1] <= args.train_range[0])
    )
    # only set epochs and/or batch_size for pytorch models
    assert bool(args.pytorch_model) is bool(args.epochs) is bool(args.batch_size)
    # only pass state_dict, optionally, for pytorch models
    assert not (
      args.sklearn_model 
      and (args.load_latest_state_dict or args.load_state_dict)
    )
    # pass either latest or specified state dict but not both
    assert not (args.load_latest_state_dict and args.load_state_dict)
    # only pass eval_metric and n_iter for sklearn models
    assert bool(args.sklearn_model) is bool(args.eval_metric) is bool(args.n_iter)