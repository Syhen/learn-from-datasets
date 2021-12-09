"""
@created by: heyao
@created at: 2021-12-08 16:26:20
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from lfd.layers import Attention


class Roberta(nn.Module):
    def __init__(self, step_dim, model_path="/Users/heyao/Desktop/roberta-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path)
        self.mean_pool = lambda x: torch.mean(x, dim=1)
        self.max_pool = lambda x: torch.max(x, dim=1)[0]
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.attention_layer = Attention(self.backbone.config.hidden_size, step_dim)
        # self.dropout = nn.Identity()
        self.head = nn.Linear(self.backbone.config.hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask)
        x = outputs.last_hidden_state
        mean_pool = self.mean_pool(x)
        attention = self.attention_layer(x)
        x = torch.cat([mean_pool, attention], dim=1)
        # print(x.shape)
        x = self.head(x)
        # print("output:", x.shape)
        return x


if __name__ == '__main__':
    import warnings

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split

    from lfd.datasets import load_disaster_tweets
    from lfd.disaster_tweets.lstm import DisasterTweetsBertDataset, train_one_epoch

    warnings.filterwarnings("ignore")
    train, test = load_disaster_tweets()

    y = train["target"].values
    X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=42, stratify=y)
    # print(pd.DataFrame({"x": [len(i) for i in X]}).describe([.5, .75, .9, .99, .997]))
    max_length = 85
    tokenizer = AutoTokenizer.from_pretrained("/Users/heyao/Desktop/roberta-base")
    dataset = DisasterTweetsBertDataset(X_train["text"].to_list(), y_train, tokenizer, max_length=max_length)
    # lengths = []
    # for x, _ ,y in dataset:
    #     lengths.append(len(x))
    # print(np.percentile(lengths, 0.997))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    val_dataset = DisasterTweetsBertDataset(X_val["text"].to_list(), y_val, tokenizer, max_length=max_length)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    model = Roberta(step_dim=max_length)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    best_score = -float("inf")
    for _ in range(3):
        score = train_one_epoch(model, dataloader, val_dataloader, criterion, optimizer, scoring="f1", verbose=1,
                                best_score=best_score, label_smoothing=0.1, validation_on_step=0)
        best_score = max(best_score, score)
