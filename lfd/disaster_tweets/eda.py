"""
@created by: heyao
@created at: 2021-12-08 14:41:44
"""
from lfd.datasets import load_disaster_tweets


train, test = load_disaster_tweets()
target = "target"
print(train[target].value_counts(normalize=True))
print(train.isnull().sum())
print(train.describe())
print(train.keyword.value_counts())
