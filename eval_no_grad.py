
import typing as T
import logging
import torch

logger = logging.getLogger(__name__)


class EvalNoGrad:
  
  def __init__(self, model: T.Type[torch.nn.Module]):
    self.model = model # models are passed by reference
    self.initModeTrain = self.model.training
    self.initGradEnabled = torch.is_grad_enabled()
    
  def __enter__(self) -> None:
    self.model.train(False)
    torch.set_grad_enabled(False)
    
  def __exit__(self, *_) -> None:
    self.model.train(self.initModeTrain)
    torch.set_grad_enabled(self.initGradEnabled)