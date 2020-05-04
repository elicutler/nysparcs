
import typing as T
import logging
import torch

logger = logging.getLogger(__name__)


class EvalNoGrad:
  '''
  Context manager to run in eval mode without gradient tracking.
  Upon exit, restore whichever train/eval and grad enabled/disabled modes
  were initially configured.
  '''
  
  def __init__(self, model) -> None:
    self.model = model # models are passed by reference
    self.startModelMode = self.model.training
    self.startGradMode = torch.is_grad_enabled()
    
  def __enter__(self) -> None:
    self.model.train(False)
    torch.set_grad_enabled(False)
    
  def __exit__(self, *args) -> None:
    self.model.train(self.startModelMode)
    torch.set_grad_enabled(self.startGradMode)