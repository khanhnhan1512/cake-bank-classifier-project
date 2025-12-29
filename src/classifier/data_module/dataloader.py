from pathlib import Path
from typing import Literal
from torch.utils.data import DataLoader

# Import class Dataset cũ của bạn
from classifier.data_module.dataset import LivenessDataset
# Import hàm transforms vừa tạo ở trên
from classifier.data_module.transform import get_transforms

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
        
        # Cấu hình phần cứng (Tối ưu cho GPU)
        # num_workers: Số luồng CPU load ảnh. Thường set = 2 hoặc 4.
        self.num_workers = 2 
        self.pin_memory = True 

    def setup(self, stage: Literal['samples', 'fit', 'test']):
        if stage == 'fit':
            # Train: Dùng transform 'train' (có Augmentation)
            self.train_dataset = LivenessDataset(
                self.train_path, 
                transform=get_transforms('train')
            )
            # Dev: Dùng transform 'val' (Chỉ Resize + Norm)
            self.dev_dataset = LivenessDataset(
                self.dev_path, 
                transform=get_transforms('val')
            )
            
        elif stage == 'test':
            # Test: Dùng transform 'val'
            self.test_dataset = LivenessDataset(
                self.test_path, 
                transform=get_transforms('val')
            )
            
        elif stage == 'samples':
            # Samples: Dùng transform 'val' để visualize chuẩn input model
            self.samples_dataset = LivenessDataset(
                self.samples_path, 
                transform=get_transforms('val')
            )

    def get_train_loader(self):
        # Train luôn cần Shuffle
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_dev_loader(self):
        # Dev KHÔNG shuffle
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
        # Samples có thể shuffle hoặc không tùy nhu cầu xem ngẫu nhiên
        return DataLoader(
            self.samples_dataset, 
            batch_size=4, # Load ít thôi để xem
            shuffle=True
        )