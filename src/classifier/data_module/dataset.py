from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class LivenessDataset(Dataset):
    LABEL_MAP = {"normal": 0, "spoof": 1}
  
    def __init__(self, data_dir: Path, transform=None):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        
        # Safety Check: Ensure directory exists
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        # Iterate through class folders (normal, spoof)
        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            if class_dir.name not in self.LABEL_MAP:
                continue
            
            label = self.LABEL_MAP[class_dir.name]
            
            # Robustness: Support multiple image file extensions
            valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
            
            for ext in valid_extensions:
                for img_path in sorted(class_dir.glob(ext)):
                    self.samples.append((img_path, label))

        # Check if no images were loaded
        if len(self.samples) == 0:
            print(f"Warning: No images found in {data_dir}. Check your path or file extensions!")

    def __len__(self):
        return len(self.samples)
  
    def __getitem__(self, index: int):
        path, label = self.samples[index]
        
        # Load image and convert to RGB
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            raise e
        
        if self.transform:
            image = self.transform(image)
        
        return image, label