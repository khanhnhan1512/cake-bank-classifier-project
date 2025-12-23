import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import mediapipe as mp # Import thư viện chuẩn

# CẤU HÌNH
RAW_DATA_DIR = './data/raw'
PROCESSED_DATA_DIR = './data/processed'
SCALE = 1.3  # Scale nhỏ lại chút để crop chặt hơn vào mặt, tránh lấy quá nhiều background thừa
IMAGE_SIZE = 224

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Hàm khởi tạo MediaPipe detection
# Tách ra ngoài để không phải init lại nhiều lần
mp_face_detection = mp.solutions.face_detection
# model_selection=1: Dùng cho ảnh chụp xa/toàn thân (robust hơn model=0 dùng cho selfie gần)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_best_face_box(pil_image):
    """
    Input: PIL Image
    Output: Box [x1, y1, x2, y2] của khuôn mặt TO NHẤT
    """
    # Convert PIL -> Numpy array (MediaPipe cần numpy)
    image_np = np.array(pil_image)
    
    # Process detection
    results = face_detection.process(image_np)
    
    if not results.detections:
        return None
        
    h_img, w_img, _ = image_np.shape
    best_box = None
    max_area = 0

    # Duyệt qua tất cả các khuôn mặt tìm được
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        
        # Convert tọa độ tương đối (0.0-1.0) sang tuyệt đối (pixels)
        x1 = int(bboxC.xmin * w_img)
        y1 = int(bboxC.ymin * h_img)
        w_box = int(bboxC.width * w_img)
        h_box = int(bboxC.height * h_img)
        
        # Sửa lỗi tọa độ âm (đôi khi mediapipe trả về âm)
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        area = w_box * h_box
        
        # LOGIC QUAN TRỌNG: Chỉ lấy box có diện tích lớn nhất
        # Người ngồi trước luôn có diện tích mặt lớn hơn người ngồi sau
        if area > max_area:
            max_area = area
            best_box = [x1, y1, x1 + w_box, y1 + h_box]

    return best_box

def preprocess_dataset():
    print("Starting Preprocessing with MediaPipe...")
    
    sets = ['train', 'dev', 'test']
    labels = ['normal', 'spoof']

    for phase in sets:
        for label in labels:
            input_dir = os.path.join(RAW_DATA_DIR, phase, label)
            output_dir = os.path.join(PROCESSED_DATA_DIR, phase, label)
            
            if not os.path.exists(input_dir):
                continue
                
            create_dir(output_dir)
            
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Processing {phase}/{label}: {len(files)} images...")
            
            no_face_count = 0
            total_count = 0

            for f in tqdm(files):
                img_path = os.path.join(input_dir, f)
                save_path = os.path.join(output_dir, f)
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    # 1. Detect bằng MediaPipe
                    box = get_best_face_box(img)
                    
                    img_final = None
                    
                    if box is not None:
                        x1, y1, x2, y2 = box
                        
                        # 2. Logic tính toán Crop + Scale (Context)
                        w = x2 - x1
                        h = y2 - y1
                        cx, cy = x1 + w//2, y1 + h//2
                        
                        # Mở rộng vùng crop theo SCALE
                        new_w = w * SCALE
                        new_h = h * SCALE
                        
                        left = max(0, int(cx - new_w/2))
                        top = max(0, int(cy - new_h/2))
                        right = min(img.width, int(cx + new_w/2))
                        bottom = min(img.height, int(cy + new_h/2))
                        
                        img_cropped = img.crop((left, top, right, bottom))
                        img_final = img_cropped.resize((IMAGE_SIZE, IMAGE_SIZE))
                    else:
                        # Fallback: Không tìm thấy mặt -> Resize ảnh gốc
                        img_final = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                        no_face_count += 1
                        # print(f"Warning: No face in {f}") # Bật lên nếu muốn debug kỹ

                    # Save
                    img_final.save(save_path)
                    total_count += 1

                except Exception as e:
                    print(f"Error {f}: {e}")
            
            print(f"-> Faces NOT detected in {no_face_count}/{total_count} images.")

if __name__ == "__main__":
    preprocess_dataset()