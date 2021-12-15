"""
@created by: heyao
@created at: 2021-12-08 14:41:44
"""
from lfd.datasets import load_disaster_tweets
from lfd.disaster_tweets.tfidf import PATTERN_URL


train, test = load_disaster_tweets()
target = "target"
print("target distribution:", train[target].value_counts(normalize=True))
print("na value:", train.isnull().sum())
print("describe:", train.describe())
print("keyword distribution:", train.keyword.value_counts())
location_count = train["location"].value_counts()
hot_location = location_count[location_count >= 10].index
print(hot_location)
print(train[train.location.isin(hot_location)].groupby("location")[target].transform("mean").value_counts())
print(train[~train.location.isin(hot_location)].groupby("location")[target].transform("mean").value_counts())
train["url_num"] = train["text"].apply(lambda x: len(PATTERN_URL.findall(x)))
print(train["url_num"].value_counts())
print(train.groupby("url_num")[target].mean())
