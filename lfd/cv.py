"""
@created by: heyao
@created at: 2021-12-08 14:44:39
"""
import pickle
from typing import AnyStr

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from lfd.utils.meters import get_scoring


def _make_cv(cv: [AnyStr, int]):
    def __make_cv(cv):
        if isinstance(cv, str) and cv.endswith(".csv"):
            df = pd.read_csv(cv)
            n_fold = int(df["fold"].max())
            for i in range(n_fold + 1):
                val_idx = df.index[df["fold"] == i]
                train_idx = df.index[df["fold"] != i]
                yield train_idx, val_idx
        if isinstance(cv, int):
            raise NotImplementedError()
    return list(__make_cv(cv))


class KFoldTrainer(object):
    def __init__(self, model_class, hyper_parameters, cv=5, fit_parameters=None, verbose=1):
        self.model_class_ = model_class
        self.hyper_parameters_ = hyper_parameters
        self.cv_ = cv
        self.fit_parameters_ = fit_parameters
        self.verbose_ = verbose
        self.kfold_ = _make_cv(cv)
        self.models = []
        self.oof = np.array([])
        self.training_predictions = []
        self.target = None

    def _fit_model(self, model, X, y, X_val, y_val, fit_parameters=None):
        fit_parameters = {} if fit_parameters is None else fit_parameters
        return model.fit(X, y, **fit_parameters)

    def fit(self, X, y=None):
        oof_pred = np.zeros((y.shape[0], ))
        self.target = y
        for train_idx, val_idx in tqdm(self.kfold_, total=len(self.kfold_), disable=not self.verbose_):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model = self.model_class_(**self.hyper_parameters_)
            model = self._fit_model(model, X_train, y_train, y_val, y_val, fit_parameters=self.fit_parameters_)
            self.models.append(model)
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)
            oof_pred[val_idx] = y_pred
            self.training_predictions.append([y_val, y_pred])
        self.oof = oof_pred
        return self.models

    def display(self, threshold=0.5, scoring="f1"):
        scoring = get_scoring(scoring)
        scores = []
        for y_val, y_pred in self.training_predictions:
            scores.append(scoring(y_val, (y_pred > threshold).astype(int)))
        print(f"cross fold, mean: {np.mean(scores):.5f}, std: {np.std(scores):.5f}")
        print(f"out of fold: {scoring(self.target, (self.oof > threshold).astype(int)):.5f}")

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

    def predict_proba(self, X):
        predictions = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                predictions.append(model.predict_proba(X)[:, 1])
            else:
                predictions.append(model.predict_proba(X)[:, 1])
        return np.mean(predictions, axis=0)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump([self.models, self.oof, self.training_predictions, self.target], f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.models, self.oof, self.training_predictions, self.target = pickle.load(f)
