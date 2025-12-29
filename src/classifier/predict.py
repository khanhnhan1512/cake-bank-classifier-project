import torch
import cv2
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
from torchvision import transforms

# Import model class
from classifier.model import LivenessDetectionModel
from classifier.preprocess import get_face_box_opencv # Tái sử dụng hàm detect mặt

# Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
SCALE = 1.6 # Quan trọng: Phải khớp với lúc preprocess train

def load_model():
    print(f"⏳ Loading model from {BEST_MODEL_PATH}...")
    model = LivenessDetectionModel(pretrained=False) # Không cần pretrained vì load weight rồi
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_single_image(image_path):
    """
    Quy trình y hệt preprocess.py nhưng cho 1 ảnh
    Detect -> Crop (Scale 1.6) -> Resize -> Transform
    """
    try:
        # 1. Load ảnh
        img_pil = Image.open(image_path).convert('RGB')
        
        # 2. Detect & Crop
        box, _ = get_face_box_opencv(img_pil) # Hàm này trả về ảnh gốc nếu không thấy mặt
        
        if box is not None:
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1
            cx, cy = x1 + w_box//2, y1 + h_box//2
            
            size = int(max(w_box, h_box) * SCALE)
            left = max(0, cx - size//2)
            top = max(0, cy - size//2)
            right = min(img_pil.width, cx + size//2)
            bottom = min(img_pil.height, cy + size//2)
            
            img_pil = img_pil.crop((left, top, right, bottom))
        
        # 3. Resize & Transform
        # Transform này phải giống hệt Val/Test transform trong dataloader.py
        val_transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = val_transforms(img_pil)
        return img_tensor.unsqueeze(0) # Thêm batch dim: [1, 3, 224, 224]
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def predict(image_path, model, threshold=0.5): # Có thể thay 0.5 bằng threshold tối ưu bạn tìm được
    img_tensor = preprocess_single_image(image_path)
    if img_tensor is None:
        return

    img_tensor = img_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        spoof_score = probs[0, 1].item() # Lấy xác suất Spoof (Class 1)
        
    print("\n" + "="*30)
    print(f"🖼️  Image: {image_path}")
    print(f"📊 Liveness Score (Spoof Probability): {spoof_score:.4f}")
    print("-" * 30)
    
    if spoof_score > threshold:
        print(f"🚨 RESULT: SPOOF (FAKE) ❌")
    else:
        print(f"✅ RESULT: REAL (LIVE)")
    print("="*30 + "\n")

if __name__ == "__main__":
    # Cách chạy: uv run src/classifier/predict.py --image "path/to/image.jpg"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()
    
    model = load_model()
    predict(args.image, model, args.threshold)