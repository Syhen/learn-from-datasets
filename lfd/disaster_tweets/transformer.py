"""
@created by: heyao
@created at: 2021-12-08 16:26:20
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import word_tokenize
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from lfd.datasets.disaster_tweets import make_cv_disaster_tweets
from lfd.disaster_tweets.lstm import fit
from lfd.layers import Attention
from lfd.utils.meters import AverageMeter, AccumulateMeter, get_scoring
from lfd.utils.pretrained import load_word_embedding


class BasicTransformer(nn.Module):
    def __init__(self, max_word, max_seq_length, word_embedding=None, embedding_dim=50,
                 d_model=50, num_head=2):
        super().__init__()
        self.embedding = nn.Embedding(max_word, embedding_dim=embedding_dim, padding_idx=0)
        if word_embedding is not None:
            self.embedding.weight.data = torch.FloatTensor(word_embedding)
            self.embedding.requires_grad_(False)
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model, num_head, dim_feedforward=d_model * num_head,
                                                              batch_first=True)
        self.mean_pool = lambda x: torch.mean(x, dim=1)
        self.max_pool = lambda x: torch.max(x, dim=1)[0]
        self.lstm = nn.LSTM(d_model, 40, batch_first=True)
        # self.attention_layer = Attention(d_model, max_seq_length)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        # self.dropout = nn.Identity()
        self.head = nn.Linear(embedding_dim * 2 + 40 * 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        # print("embedding:", x.shape)
        x = self.transformer_encoder(x)
        # print("transformer:", x.shape)
        lstm_out, _ = self.lstm(x)
        # print("lstm:", x.shape)
        mean_pooling = self.flatten(self.mean_pool(x))
        max_pooling = self.flatten(self.max_pool(x))
        mean_pooling_lstm = self.flatten(self.mean_pool(lstm_out))
        # attention = self.attention_layer(x)
        x = torch.cat([mean_pooling, max_pooling, mean_pooling_lstm], dim=1)
        # print("before head:", x.shape)
        x = self.dropout(x)
        x = self.head(x)
        # print("output:", x.shape)
        return x


if __name__ == '__main__':
    # mean: 0.78216, std: 0.00843 (nhead=2)
    # mean: 0.78551, std: 0.01370 (nhead=2, d_model=100)
    # mean: 0.78631, std: 0.01035 (add lstm 20)
    # mean: 0.78632, std: 0.00838 (add lstm 40)
    import warnings

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split

    from lfd.datasets import load_disaster_tweets
    from lfd.utils.reproduction import seed_everything

    warnings.filterwarnings("ignore")
    train, test = load_disaster_tweets()

    y = train["target"].values
    tokenizer = Tokenizer(num_words=None, lower=True)
    tokenizer.fit_on_texts(pd.concat([train, test], axis=0)["text"])
    X = tokenizer.texts_to_sequences(train["text"])
    X = pad_sequences(X, maxlen=33)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test = tokenizer.texts_to_sequences(test["text"])
    X_test = pad_sequences(X_test, maxlen=33)
    # print(pd.DataFrame({"x": [len(i) for i in X]}).describe([.5, .75, .9, .99, .997]))
    criterion = nn.BCEWithLogitsLoss()
    model_class = BasicTransformer
    embedding_dim = 100
    model_parameters = dict(
        max_word=len(tokenizer.word_index) + 1, max_seq_length=33,
        d_model=embedding_dim, num_head=2, embedding_dim=embedding_dim
    )
    word_embedding = load_word_embedding(f"/Users/heyao/Downloads/glove.6B.{embedding_dim}d.txt", tokenizer.word_index,
                                         vector_size=embedding_dim)
    print(word_embedding.shape)
    model_parameters["word_embedding"] = word_embedding
    seed_everything(42)
    test_pred = fit(model_class, model_parameters, X, y, X_test, criterion, epochs=30, label_smoothing=0.1)
    df_submit = pd.DataFrame()
    df_submit["id"] = test["id"]
    df_submit["target"] = (test_pred > 0.5).astype(int)
    df_submit.to_csv("/Users/heyao/Desktop/transformer.csv", index=False)
