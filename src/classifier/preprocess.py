import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps
import mediapipe as mp

# CONFIGURATION 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Crop settings
SCALE = 1.6       
IMAGE_SIZE = 224  # Output size for EfficientNet/MobileNet

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_face_box_robust(pil_image):
    """
    Try detecting at 0 degrees. If failed -> Rotate 90 -> 180 -> 270.
    """
    rotations = [0, 90, 180, 270]
    
    # try:
    #     pil_image = ImageOps.exif_transpose(pil_image)
    # except Exception:
    #     pass

    for angle in rotations:
        if angle == 0:
            current_img = pil_image
        else:
            # expand=True ensures the image is not cropped during rotation
            current_img = pil_image.rotate(-angle, expand=True)
            
        image_np = np.array(current_img)
        results = face_detection.process(image_np)
        
        if results.detections:
            h_img, w_img, _ = image_np.shape
            max_area = 0
            best_box = None
            
            # Find the largest face 
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * w_img)
                y1 = int(bboxC.ymin * h_img)
                w_box = int(bboxC.width * w_img)
                h_box = int(bboxC.height * h_img)
                
                # Fix negative coordinates
                x1, y1 = max(0, x1), max(0, y1)
                
                area = w_box * h_box
                if area > max_area:
                    max_area = area
                    best_box = [x1, y1, x1 + w_box, y1 + h_box]
            
            if best_box is not None:
                return best_box, current_img
                
    # If no face found after all rotations, return None
    return None, pil_image

def process_image(img_path, save_path):
    """Process a single image: Detect -> Crop -> Resize -> Save"""
    try:
        img = Image.open(img_path).convert('RGB')
        
        box, correct_img = get_face_box_robust(img)
        
        if box is not None:
            # Calculate Square Crop
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1
            cx, cy = x1 + w_box//2, y1 + h_box//2
            
            # Scale up to capture context
            size = int(max(w_box, h_box) * SCALE)
            
            left = max(0, cx - size//2)
            top = max(0, cy - size//2)
            right = min(correct_img.width, cx + size//2)
            bottom = min(correct_img.height, cy + size//2)
            
            img_cropped = correct_img.crop((left, top, right, bottom))
            img_final = img_cropped.resize((IMAGE_SIZE, IMAGE_SIZE))
            
            # Save
            img_final.save(save_path, quality=95)
            return True 
        else:
            # Resize original image if detection fails
            img_final = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_final.save(save_path, quality=95)
            return False
            
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def main():
    print("Starting Data Preprocessing...")
    print(f"Input Directory:  {RAW_DATA_DIR}")
    print(f"Output Directory: {PROCESSED_DATA_DIR}")

    # Remove old processed folder to ensure a clean state
    if os.path.exists(PROCESSED_DATA_DIR):
        print("Removing old processed directory...")
        shutil.rmtree(PROCESSED_DATA_DIR)
    
    # Count total files 
    total_files = 0
    success_count = 0
    
    # Recursively find all image files
    for file_path in RAW_DATA_DIR.rglob("*"):
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            total_files += 1

    print(f"Total images found: {total_files}")
    
    with tqdm(total=total_files, desc="Processing") as pbar:
        for file_path in RAW_DATA_DIR.rglob("*"):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                relative_path = file_path.relative_to(RAW_DATA_DIR)
                output_path = PROCESSED_DATA_DIR / relative_path
                
                # Create parent directory if needed
                create_dir(output_path.parent)
                
                # Process image
                is_face_detected = process_image(file_path, output_path)
                
                if is_face_detected:
                    success_count += 1
                
                pbar.update(1)
                
    print("-" * 40)
    print("Processing Completed!")
    if total_files > 0:
        print(f"Successful Detections: {success_count}/{total_files} ({success_count/total_files*100:.1f}%)")
    print(f"Data saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()