from pathlib import Path
from typing import Literal
from torchvision.transforms import Compose # type: ignore
from torch.utils.data import Dataset, DataLoader

from classifier.data_module.dataset import LivenessDataset

class LivenessDataLoader:
  def __init__(self, train_path: Path,
                dev_path: Path,
                test_path: Path,
                samples_path: Path, 
                transform: Compose | None=None,
                batch_size=32,
                shuffle=True):
      self.samples_path = samples_path
      self.train_path = train_path
      self.dev_path = dev_path
      self.test_path = test_path
      self.transform = transform
      self.batch_size = batch_size
      self.shuffle = shuffle
      
  def get_dataloader(self, dataset: Dataset) -> DataLoader:
      return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
  def setup(self, stage: Literal['samples','fit', 'test']):
      if stage == 'fit':
          self.train_dataset = LivenessDataset(self.train_path, self.transform)
          self.dev_dataset = LivenessDataset(self.dev_path, self.transform)
          
      elif stage == 'samples':
          self.samples_dataset = LivenessDataset(self.samples_path, self.transform)
      elif stage == 'test':
          self.test_dataset = LivenessDataset(self.test_path, self.transform)
          
  def get_train_loader(self):
      return self.get_dataloader(self.train_dataset)
    
  def get_dev_loader(self):
      return self.get_dataloader(self.dev_dataset)
    
  def get_samples_loader(self):
      return self.get_dataloader(self.samples_dataset)
    
  def get_test_loader(self):
      return self.get_dataloader(self.test_dataset)