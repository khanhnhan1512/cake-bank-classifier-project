from pathlib import Path
from typing import Literal
from torch.utils.data import DataLoader
import torch
import numpy as np
import random

from classifier.data_module.dataset import LivenessDataset
from classifier.data_module.transform import get_transforms

def seed_worker(worker_id):
    """Seed generator for individual workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class LivenessDataLoader:
    def __init__(self, 
                 train_path: Path,
                 dev_path: Path,
                 test_path: Path,
                 samples_path: Path, 
                 batch_size=32):
        
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.samples_path = samples_path
        self.batch_size = batch_size
        self.num_workers = 2 
        self.pin_memory = True 

    def setup(self, stage: Literal['samples', 'fit', 'test']):
        if stage == 'fit':
            self.train_dataset = LivenessDataset(
                self.train_path, 
                transform=get_transforms('train')
            )
            self.dev_dataset = LivenessDataset(
                self.dev_path, 
                transform=get_transforms('val')
            )
            
        elif stage == 'test':
            self.test_dataset = LivenessDataset(
                self.test_path, 
                transform=get_transforms('val')
            )
            
        elif stage == 'samples':
            self.samples_dataset = LivenessDataset(
                self.samples_path, 
                transform=get_transforms('val')
            )

    def get_train_loader(self):
        g = torch.Generator()
        g.manual_seed(42)
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_dev_loader(self):
        g = torch.Generator()
        g.manual_seed(42)
        return DataLoader(
            self.dev_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_test_loader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_samples_loader(self):
        g = torch.Generator()
        g.manual_seed(42)
        return DataLoader(
            self.samples_dataset, 
            batch_size=4,
            shuffle=True
        )