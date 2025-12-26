from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    
def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, pos_label=1)
    metrics['recall'] = recall_score(y_true, y_pred, pos_label=1)
    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    metrics['avg_precision'] = average_precision_score(y_true, y_prob)
    metrics['f1_score'] = f1_score(y_true, y_pred, pos_label=1)
    return metrics