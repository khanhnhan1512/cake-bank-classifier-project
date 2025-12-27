from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose # type: ignore
from torchvision import transforms # type: ignore

class LivenessDataset(Dataset):
  LABEL_MAP = {"normal": 0, "spoof": 1}
  
  def __init__(self, data_dir: Path, 
                transform: Compose | None=None):
    self.samples: list[tuple[Path, int]] = []
    self.transform = transform
    
    for class_dir in sorted(data_dir.iterdir()):
      if not class_dir.is_dir():
        continue
      
      label = self.LABEL_MAP[class_dir.name]
      for img_path in sorted(class_dir.glob("*.jpg")):
        self.samples.append((img_path, label))
    

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, index: int):
    path, label = self.samples[index]
    image = Image.open(path).convert("RGB")
    
    if self.transform:
      image = self.transform(image)
    else:
      transform = Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor()
      ])
      image = transform(image)
    
    return image, label

