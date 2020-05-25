
import typing as T
import logging 
import argparse

from constants import EVAL_OPTIONS

logger = logging.getLogger(__name__)


class DeployArgParser:
  
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
    self.parser.add_argument(
      '--model_name', type=str, help=(
        'Deploy model with given name (exclude file type extension), '
        ' i.e. path stem'
      )
    )
    self.parser.add_argument(
      '--best_model', action='store_true', help=(
        'Deploy best model. If specified, must also provide target and eval_metric.'
      )
    )
    self.parser.add_argument(
      '--target', type=str, help='target variable for selecting best model.'
    )
    self.parser.add_argument(
      '--eval_metric', type=str, help=(
        'Evaluation metric for selecting best model.'
      )
    )
    
  def parseArgs(self) -> argparse.Namespace:
    args = self.parser.parse_args()
    self._validateArgs(args)
    return args
  
  def _validateArgs(self, args) -> None:
    
    # Deploy model by name XOR best model
    assert bool(args.model_name) + bool(args.best_model) == 1
    # If deploying best model, must specify target and eval metric
    assert bool(args.best_model) is bool(args.target) is bool(args.eval_metric)
    # If deploying best_model by eval_metric, must be valid metric
    if args.eval_metric is not None:
      assert args.eval_metric in EVAL_OPTIONS
