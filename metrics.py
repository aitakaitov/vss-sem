import numpy as np


def evaluate_predictions(y_score, y_true) -> dict:
    y_pred = np.argmax(y_score, axis=1)
    tp, tn, fp, fn = count(y_true, y_pred)
    results = {
        'precision': precision_score(tp, fp),
        'recall': recall_score(tp, fn),
        'f1': f1_score(tp, fp, fn),
        'fa': fa_score(tn, fp, fn)
    }

    return results


def count(y_true, y_pred):
    fp = 0.0
    fn = 0.0
    tp = 0.0
    tn = 0.0

    for pred, true in zip(y_pred, y_true):
        if pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
        elif pred == 0 and true == 0:
            tn += 1
        elif pred == 1 and true == 1:
            tp += 1

    return tp, tn, fp, fn


def fa_score(tn, fp, fn):
    return fn / (fp + tn)


def f1_score(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn)


def recall_score(tp, fn):
    return tp / (tp + fn)


def precision_score(tp, fp):
    return tp / (tp + fp)
