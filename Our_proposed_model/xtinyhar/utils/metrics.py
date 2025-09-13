
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    return {"acc": acc, "f1": f1, "kappa": kappa}
