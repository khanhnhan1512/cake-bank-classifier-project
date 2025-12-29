import torch
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
from torchvision import transforms
import sys

current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

# Import model & preprocess
try:
    from classifier.model import LivenessDetectionModel
    from classifier.preprocess import get_face_box_robust 
except ImportError:
    from model import LivenessDetectionModel
    from preprocess import get_face_box_robust

# Config
PROJECT_ROOT = src_dir.parent
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
SCALE = 1.6 

def load_model():
    print(f"Loading model from {BEST_MODEL_PATH}...")
    if not BEST_MODEL_PATH.exists():
        print(f"Error: Model file not found at {BEST_MODEL_PATH}")
        sys.exit(1)
        
    model = LivenessDetectionModel(pretrained=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict(image_path, model, threshold=0.5):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        return

    try:
        img_pil = Image.open(img_path).convert('RGB')
        
        #Detect & Crop (Using MediaPipe function)
        box, corrected_img = get_face_box_robust(img_pil)
        
        if box is not None:
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1
            cx, cy = x1 + w_box//2, y1 + h_box//2
            
            size = int(max(w_box, h_box) * SCALE)
            left = max(0, cx - size//2)
            top = max(0, cy - size//2)
            right = min(corrected_img.width, cx + size//2)
            bottom = min(corrected_img.height, cy + size//2)
            
            img_pil = corrected_img.crop((left, top, right, bottom))
        else:
            print("Warning: No face detected. Using full image.")

        # Transform
        val_transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = val_transforms(img_pil).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            spoof_score = probs[0, 1].item()
            
        print("\n" + "="*30)
        print(f"Image: {img_path.name}")
        print(f"Liveness Score: {spoof_score:.4f}")
        print("-" * 30)
        
        if spoof_score > threshold:
            print(f"RESULT: SPOOF (FAKE)")
        else:
            print(f"RESULT: REAL (LIVE)")
        print("="*30 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()
    
    model = load_model()
    predict(args.image, model, args.threshold)