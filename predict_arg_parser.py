
import typing as T
import logging 
import argparse

from constants import EVAL_OPTIONS

logger = logging.getLogger(__name__)


class PredictArgParser:
  
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
    self.parser.add_argument(
      '--model_name', type=str, help=(
        'Load model with given name (exclude file type extension), '
        ' i.e. path stem'
      )
    )
    self.parser.add_argument(
      '--best_model', action='store_true', help=(
        'Load best model. If specified, must also provide target and eval_metric.'
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
    self.parser.add_argument(
      '--instances', type=str, help=(
        'Instances to predict on. Either JSON file or JSON string'
        ' (with outer quotation marks and escaped inner quotation marks).'
      )
    )
    
  def parseArgs(self) -> argparse.Namespace:
    args = self.parser.parse_args()
    self._validateArgs(args)
    return args
  
  def _validateArgs(self, args: argparse.Namespace) -> None:
    
    # Load model by name XOR best model
    assert bool(args.model_name) + bool(args.best_model) == 1
    # If loading best model, must specify target and eval metric
    assert bool(args.best_model) is bool(args.target) is bool(args.eval_metric)
    # If loading best_model by eval_metric, must be valid metric
    if args.eval_metric is not None:
      assert args.eval_metric in EVAL_OPTIONS
    # Need instances
    assert args.instances is not None

