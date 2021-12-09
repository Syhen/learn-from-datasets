"""
@created by: heyao
@created at: 2021-12-08 14:39:47
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from nltk import word_tokenize
from nltk.stem import PorterStemmer

from lfd.datasets import load_disaster_tweets
from lfd.cv import KFoldTrainer
from lfd.datasets.disaster_tweets import make_cv_disaster_tweets


stemmer = PorterStemmer()


def clean(text):
    text = text.lower()
    if "#" in text:
        text = text.replace("#", "")
    return text


def split(text):
    return " ".join(text.split())


def word_split(text):
    return " ".join(w for w in word_tokenize(text))


def tfidf_feature(train, test):
    train["text"] = train["text"].apply(clean)
    test["text"] = test["text"].apply(clean)
    # feature = pd.concat([train, test], axis=0)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5)
    tfidf_vectorizer.fit(train["text"].apply(word_split))
    X = tfidf_vectorizer.transform(train["text"]).todense()
    X_test = tfidf_vectorizer.transform(test["text"].apply(word_split)).todense()
    y = train["target"]
    return X, X_test, y


def keyword_feature(train, test):
    encoder = OneHotEncoder()
    encoder.fit(train["keyword"].values.reshape(-1, 1))
    # keyword_ratio = train["keyword"].fillna("no-cat").value_counts(normalize=True).to_dict()
    train_k = np.array(encoder.transform(train["keyword"].values.reshape(-1, 1)).todense())
    test_k = np.array(encoder.transform(test["keyword"].values.reshape(-1, 1)).todense())
    # train_ratio = train["keyword"].fillna("no-cat").map(keyword_ratio).values
    # print(train_ratio[0])
    # for i in range(len(train_k)):
    #     train_k[i, train_k[i, :] == 1.0] = train_ratio[i]
    # train_k[train_k == 1.0] = train["keyword"].map(keyword_ratio).values
    # test_k[train_k == 1.0] = test["keyword"].map(keyword_ratio).values
    return train_k, test_k


def train_model(X, y, X_test, name):
    fold_filename = make_cv_disaster_tweets(X, y, cv=5)
    model_class = linear_model.LogisticRegression
    trainer = KFoldTrainer(model_class, {}, cv=fold_filename)
    trainer.fit(X, y)
    trainer.display(threshold=0.5, scoring="f1")
    test_pred = trainer.predict(X_test)
    df_submit = pd.DataFrame()
    df_submit["id"] = test["id"]
    df_submit["target"] = test_pred
    df_submit.to_csv(f"/Users/heyao/Desktop/{name}.csv", index=False)


def tfidf_baseline(train, test):
    # 0.791 online
    # cross fold, mean: 0.74093, std: 0.00847
    # out of fold: 0.74095
    X, X_test, y = tfidf_feature(train, test)
    train_model(X, y, X_test, name="submission-tfidf-lr")


def tfidf_plus_keyword(train, test):
    #
    # cross fold, mean: 0.75189, std: 0.01375
    # out of fold: 0.75175
    X, X_test, y = tfidf_feature(train, test)
    X_keywords, X_test_keywords = keyword_feature(train, test)
    X = np.concatenate([X, X_keywords], axis=1)
    # print(X[0, :].tolist())
    X_test = np.concatenate([X_test, X_test_keywords], axis=1)
    train_model(X, y, X_test, name="submission-tfidf-keyword-lr")


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    train, test = load_disaster_tweets()
    tfidf_plus_keyword(train, test)
    # tfidf_baseline(train, test)
