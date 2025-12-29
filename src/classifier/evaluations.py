from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def compute_metrics(y_true, y_pred, y_prob):
    """
    y_true: nhãn thực tế [0, 1, 0...]
    y_pred: nhãn dự đoán [0, 1, 1...]
    y_prob: xác suất lớp 1 (spoof) [0.1, 0.9, 0.8...]
    """
    metrics = {}
    
    # Các metric cơ bản
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Cần try-catch vì nếu y_true chỉ có 1 class duy nhất thì ROC_AUC sẽ lỗi
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['avg_precision'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['roc_auc'] = 0.5
        metrics['avg_precision'] = 0.0
        
    return metrics