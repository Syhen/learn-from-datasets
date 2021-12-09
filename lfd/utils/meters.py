"""
@created by: heyao
@created at: 2021-12-08 18:03:27
"""
from lfd.metrics import SCORES


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccumulateMeter(object):
    def __init__(self, func, **kwargs):
        self.reset()
        self.func = func
        self.kwargs = kwargs

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.target = []
        self.prediction = []

    def update(self, prediction, target):
        self.prediction.extend((prediction.detach().cpu().numpy() > .5).astype(int).reshape(-1, ).tolist())
        self.target.extend(target.cpu().numpy().reshape(-1, ).tolist())
        self.avg = self.func(self.target, self.prediction, **self.kwargs)


def get_scoring(scoring):
    if scoring not in SCORES:
        raise ValueError(f"scoring {scoring} not found. Choose one from {', '.join(SCORES.keys())}")
    scoring = SCORES[scoring]
    return scoring
