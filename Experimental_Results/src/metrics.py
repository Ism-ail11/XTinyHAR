import numpy as np

def confusion_matrix(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes is None:
        n_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return (y_true == y_pred).mean()

def _prec_rec_f1_from_cm(cm):
    # per-class precision/recall/F1
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp == 0, 0.0, tp / (tp + fp))
        recall    = np.where(tp + fn == 0, 0.0, tp / (tp + fn))
        f1 = np.where(precision + recall == 0, 0.0, 2 * precision * recall / (precision + recall))
    return precision, recall, f1

def prf_macro(y_true, y_pred, n_classes=None):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    p, r, f1 = _prec_rec_f1_from_cm(cm)
    return p.mean(), r.mean(), f1.mean()

def prf_micro(y_true, y_pred, n_classes=None):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).sum().astype(float)
    fp = cm.sum(axis=0).sum() - np.diag(cm).sum()
    fn = cm.sum(axis=1).sum() - np.diag(cm).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def cohens_kappa(y_true, y_pred, n_classes=None):
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    n = cm.sum()
    po = np.trace(cm) / n if n > 0 else 0.0
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n ** 2) if n > 0 else 0.0
    return 0.0 if (1 - pe) == 0 else (po - pe) / (1 - pe)
