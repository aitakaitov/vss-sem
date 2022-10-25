from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import numpy as np


def evaluate_predictions(y_score, y_true) -> dict:
    y_score_greater = np.array(y_score)[:, 1]
    y_pred = np.argmax(y_score, axis=1)
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': 1 - precision_score(y_true, y_pred),
        'recall': 1 - recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='micro'),
        'auc': roc_auc_score(y_true, y_score_greater, average='micro'),
        'fa': fa_score(y_pred, y_true)
    }

    return results


def fa_score(y_pred, y_true):
    fp = 0.0
    fn = 0.0
    tn = 0.0
    for pred, true in zip(y_pred, y_true):
        if pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
        elif pred == 0 and true == 0:
            tn += 1

    return fn / (fp + tn)
