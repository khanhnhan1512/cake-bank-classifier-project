import numpy as np
import torch

class EarlyStopping:
    """Dừng training sớm nếu validation loss không giảm sau một khoảng patience."""
    def __init__(self, patience=7, verbose=False, delta=0.0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss # Chuyển loss thành score (càng lớn càng tốt để dễ so sánh)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model) # <--- QUAN TRỌNG: Lưu lần đầu
        elif score < self.best_score + self.delta:
            # Loss không giảm đủ nhiều (Score không tăng)
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Loss giảm tốt -> Reset counter và LƯU MODEL
            self.best_score = score
            self.save_checkpoint(val_loss, model) # <--- QUAN TRỌNG: Lưu khi cải thiện
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Lưu model khi validation loss giảm.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # Lưu state_dict (weight) thay vì cả model
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss