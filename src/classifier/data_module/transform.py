from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224) 

def get_transforms(phase: str):
    """
    Trả về transform pipeline tương ứng với phase: 'train' hoặc 'val'
    """
    if phase == 'train':
        return transforms.Compose([
            # 1. Geometric
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            
            # 2. Photometric (Chỉ train mới dùng)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            
            # 3. Standardize
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            
            # 4. Regularization
            transforms.RandomErasing(p=0.1) 
        ])
    
    else: # 'val', 'test', 'samples'
        return transforms.Compose([
            transforms.Resize(256),            
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])