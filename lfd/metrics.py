"""
@created by: heyao
@created at: 2021-12-08 15:15:10
"""
from sklearn import metrics as sk_metrics


SCORES = {
    "f1_score": sk_metrics.f1_score,
    "f1": sk_metrics.f1_score,
    "acc": sk_metrics.accuracy_score,
    "accuracy_score": sk_metrics.accuracy_score,
    "accuracy": sk_metrics.accuracy_score,
}
