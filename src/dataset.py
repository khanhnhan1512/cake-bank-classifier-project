import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    """
    Hàm này trả về 3 dataloaders: train, dev, test
    """
    
    # 1. Định nghĩa Augmentation (Chỉ áp dụng cho Train)
    train_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(), # Có thể bật nếu muốn
        # transforms.ColorJitter(brightness=0.2, contrast=0.2), # Giả lập ánh sáng thay đổi
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn ImageNet
    ])

    # Dev/Test chỉ cần Normalize, giữ nguyên ảnh
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Data từ folder processed
    # Cấu trúc folder chuẩn nên dùng ImageFolder là tối ưu nhất
    
    # Train Loader
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        train_loader = None
        
    # Dev Loader
    dev_dir = os.path.join(data_dir, 'dev')
    if os.path.exists(dev_dir):
        dev_dataset = datasets.ImageFolder(dev_dir, transform=val_transforms)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        dev_loader = None

    # Test Loader
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        test_loader = None
        
    return train_loader, dev_loader, test_loader

# Test logic
if __name__ == "__main__":
    # Test thử xem load được không
    print("Checking Dataloader...")
    t_loader, d_loader, _ = get_dataloaders('./data/processed')
    if t_loader:
        images, labels = next(iter(t_loader))
        print(f"Train batch shape: {images.shape}") # Mong đợi: [32, 3, 224, 224]
        print(f"Labels: {labels}")