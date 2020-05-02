
import typing as T
import logging
import pandas as pd

from abc import abstractmethod
from overrides import EnforceOverrides, overrides, final
from torch.utils.data import Dataset
from torch import Tensor

logger = logging.getLogger(__name__)


class TorchDataset(Dataset):
  
  def __init__(self, params) -> None:
    self.params = params.copy()
    
  def loadDF(self, inDF) -> None:
    self.df = inDF.copy()
    
  def loadSKLearnProcessor(self, sklearnProcessor) -> None:
    self.sklearnProcessor = sklearnProcessor
    
  def validateFeatures(self) -> None:
    assert (
      (self.df.drop(columns=self.params['target']).columns 
      == self.sklearnProcessor.features)
    ).all()
    
  def __len__(self) -> int:
    return self.df.shape[0]
  
  def __getitem__(self, idx) -> T.Tuple[Tensor]:
    featureInputs = self.df.drop(columns=self.params['target']).iloc[idx]
    X = self.sklearnProcessor.transform(featureInputs)
    y = self.df['target'].iloc[idx]
    breakpoint()
  
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample