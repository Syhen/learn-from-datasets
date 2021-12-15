"""
@created by: heyao
@created at: 2021-12-08 14:33:28
"""
import os.path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from lfd import settings
from lfd.datasets.util import init_path


def load_disaster_tweets(path=None):
    dataset_path = path or settings.PATH_DATASETS
    dataset_path = init_path(dataset_path)
    dataset_name = settings.NAME_DISASTER_TWEETS
    train = pd.read_csv(dataset_path / dataset_name / "train.csv")
    test = pd.read_csv(dataset_path / dataset_name / "test.csv")
    return train, test


def make_cv_disaster_tweets(X, y, cv=5, force=False, return_filename=False):
    def _index_to_folds(fold_df):
        n_folds = fold_df.fold.nunique()
        for fold in range(n_folds):
            val_idx = fold_df[fold_df.fold == fold].index.values
            train_idx = fold_df[fold_df.fold != fold].index.values
            yield train_idx, val_idx

    filename = str(settings.PATH_DATASETS / f"{settings.NAME_DISASTER_TWEETS}/fold-{cv}.csv")
    if os.path.isfile(filename) and not force:
        if return_filename:
            return filename
        return list(_index_to_folds(pd.read_csv(filename)))
    kfold = StratifiedKFold(cv, shuffle=True, random_state=42)
    df = pd.DataFrame({"target": y})
    for i, (_, val_idx) in enumerate(kfold.split(X, y)):
        df.loc[val_idx, "fold"] = i
    df[["fold"]].to_csv(filename, index=False)
    if return_filename:
        return filename
    return list(_index_to_folds(df))
