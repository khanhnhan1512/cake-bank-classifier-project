from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class LivenessDataset(Dataset):
    LABEL_MAP = {"normal": 0, "spoof": 1}
  
    def __init__(self, data_dir: Path, transform=None):
        """
        Args:
            data_dir (Path): Đường dẫn đến folder chứa data (train/dev/test)
            transform (callable, optional): Transform pipeline (được truyền từ DataLoader)
        """
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        
        # 1. Safety Check: Đảm bảo đường dẫn tồn tại
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        # 2. Duyệt qua các folder class (normal, spoof)
        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            # Chỉ xử lý các folder có tên trong LABEL_MAP
            if class_dir.name not in self.LABEL_MAP:
                continue
            
            label = self.LABEL_MAP[class_dir.name]
            
            # 3. Robustness: Hỗ trợ nhiều đuôi file ảnh khác nhau
            # Đôi khi dataset có lẫn file .png hoặc .jpeg
            valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
            
            for ext in valid_extensions:
                # Dùng glob để tìm file, sorted để đảm bảo thứ tự nhất quán
                for img_path in sorted(class_dir.glob(ext)):
                    self.samples.append((img_path, label))

        # Check nếu không load được ảnh nào
        if len(self.samples) == 0:
            print(f"Warning: No images found in {data_dir}. Check your path or file extensions!")

    def __len__(self):
        return len(self.samples)
  
    def __getitem__(self, index: int):
        path, label = self.samples[index]
        
        # 4. Load ảnh và convert sang RGB (đề phòng ảnh grayscale hoặc RGBA gây lỗi model)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Trong thực tế, có thể return None và xử lý ở collate_fn, 
            # nhưng đơn giản nhất là crash để bạn biết mà sửa data rác.
            raise e
        
        # Áp dụng transform được truyền vào từ bên ngoài
        if self.transform:
            image = self.transform(image)
        
        return image, label