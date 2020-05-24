
import typing as T
import logging 
import argparse

logger = logging.getLogger(__name__)


class DeployArgParser:
  
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
    self.parser.add_argument(
      '--model_name', type=str, help=(
        'Deploy model with given name (exclude file type extension)'
      )
    )
    self.parser.add_argument(
      '--best_model_by_metric', type=str, help=(
        'Deploy model with best performance on validation set'
        ' based on specified metric.'
      )
    )
    
  def parseArgs(self) -> argparse.Namespace:
    args = self.parser.parse_args()
    self._validateArgs(args)
    return args
  
  def _validateArgs(self, args) -> None:
    
    assert (
      bool(args.model_name) + bool(args.best_model_by_metric) == 1
    )
    
    assert args.best_model_by_metric in [
      None, 'roc_auc', 'pr_auc', 'mae', 'mse'
    ]
