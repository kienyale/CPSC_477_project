from typing import List, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def compute_metrics(
    predictions: List[int],
    labels: List[int],
    average: str = 'binary'
) -> Dict[str, float]:
    """
    computes classification metrics
    
    args:
    - predictions: model predictions (0/1)
    - labels: true labels (0/1)
    - average: how to average multi-class
    """
    # convert to numpy
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # basic metrics
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average=average, zero_division=0),
        'recall': recall_score(labels, predictions, average=average, zero_division=0),
        'f1': f1_score(labels, predictions, average=average, zero_division=0)
    }
    
    # roc auc if both classes present
    if len(np.unique(labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(labels, predictions)
    
    # confusion matrix stats
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })
    
    # derived metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # negative predictive value
    
    return metrics

if __name__ == "__main__":
    # quick test
    predictions = [1, 0, 1, 1, 0, 1, 0, 1]
    labels = [1, 0, 1, 0, 0, 1, 1, 1]
    
    metrics = compute_metrics(predictions, labels)
    print("Classification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 