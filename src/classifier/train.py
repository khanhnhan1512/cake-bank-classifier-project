import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import os

# --- IMPORTS TƯƠNG ĐỐI (Relative Imports) ---
# Import các module nằm cùng thư mục hoặc thư mục con
from model import LivenessDetectionModel
from data_module.dataloader import LivenessDataLoader
from evaluations import compute_metrics
from regularizations import EarlyStopping

# --- CẤU HÌNH (CONFIG) ---
# Lấy đường dẫn gốc dự án (đi ngược lên 2 cấp từ file này: src/classifier/ -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# QUAN TRỌNG: Dùng data đã qua xử lý (processed)
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = DATA_DIR / "train"
DEV_DIR = DATA_DIR / "dev"
TEST_DIR = DATA_DIR / "test"
SAMPLE_DIR = PROJECT_ROOT / "data" / "raw" / "samples" # Sample để raw cũng được

# Nơi lưu model
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ETA_MIN = 1e-4


def seed_everything(seed=42):
    """
    Khóa mọi nguồn ngẫu nhiên để đảm bảo kết quả tái lập được (Reproducible).
    """
    # 1. Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Nếu dùng multi-GPU
    
    # 4. CUDNN (Quan trọng nhất để GPU chạy giống nhau)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- HÀM TRAIN ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.long().to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return total_loss / len(loader)

# --- HÀM EVALUATE ---
@torch.no_grad()
def evaluate(model, loader, criterion, stage="Val"):
    model.eval()
    total_loss = 0.0
    
    all_preds, all_labels, all_probs = [], [], []
    
    pbar = tqdm(loader, desc=stage, leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.long().to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # Tính toán metric
        probs = torch.softmax(outputs, dim=1)[:, 1] # Xác suất class 1 (Spoof)
        preds = torch.argmax(outputs, dim=1)
        # preds = (probs[:, 1] > 0.85).long()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    return avg_loss, metrics

# --- MAIN ---
def main():

    seed_everything(42)

    print(f"🚀 Starting training on device: {DEVICE}")
    print(f"📂 Data Directory: {DATA_DIR}")

    # 1. Setup Data
    data_loader = LivenessDataLoader(
        train_path=TRAIN_DIR,
        dev_path=DEV_DIR,
        test_path=TEST_DIR,
        samples_path=SAMPLE_DIR,
        batch_size=BATCH_SIZE
    )
    print("⏳ Loading datasets...")
    data_loader.setup('fit')
    train_loader = data_loader.get_train_loader()
    dev_loader = data_loader.get_dev_loader()
    print("✅ Data loaded successfully.")

    # 2. Setup Model
    model = LivenessDetectionModel(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=ETA_MIN)
    early_stopping = EarlyStopping(patience=5, delta=0.0, path=str(BEST_MODEL_PATH), verbose=True)

    # 3. Training Loop
    print("\nBegin Training Loop:")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 20)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_metrics = evaluate(model, dev_loader, criterion, stage="Val")
        
        # Scheduler Step
        scheduler.step()
        
        # Print Result
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")
        
        # Early Stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break

    # 4. Test Phase
    print("\n" + "="*30)
    print("🏆 Training Finished. Running Test on Best Model...")
    
    # Load lại weight tốt nhất
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    
    # Setup test data
    data_loader.setup('test')
    test_loader = data_loader.get_test_loader()
    
    test_loss, test_metrics = evaluate(model, test_loader, criterion, stage="Test")
    
    print(f"FINAL TEST RESULTS (Loss: {test_loss:.4f})")
    for k, v in test_metrics.items():
        print(f" - {k}: {v:.4f}")

if __name__ == "__main__":
    main()