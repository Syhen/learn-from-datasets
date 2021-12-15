"""
@created by: heyao
@created at: 2021-12-10 15:09:22
"""
from sklearn.metrics import f1_score


def find_best_threshold(prediction, target, scoring=f1_score, threshold_range=(0, 1000), threshold_sub=1000):
    scores = []
    for threshold in range(*threshold_range):
        threshold /= threshold_sub
        score = scoring(target, (prediction >= threshold).astype(int))
        scores.append((threshold, score))
    return scores
