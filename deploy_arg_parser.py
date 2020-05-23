
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
      '--best_model', action='store_true', help=(
        'Deploy model with best performance on validation set.'
      )
    )
    self.parser.add_argument(
      '--model_type', type=str, help=(
        'If specified, must be either pytorch or sklearn.'
        ' Required when selecting model by name,'
        ' optional when selecting best_model.'
      )
    )
    
  def parseArgs(self) -> argparse.Namespace:
    args = self.parser.parse_args()
    self._validateArgs(args)
    return args
  
  def _validateArgs(self, args) -> None:
    
    assert bool(args.model_name) + bool(args.best_model) == 1
    
    if bool(args.model_name):
      assert args.model_type in ['pytorch', 'sklearn']
      
    if bool(args.best_model):
      assert args.model_type in [None, 'pytorch', 'sklearn']
