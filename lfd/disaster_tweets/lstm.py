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

from lfd import settings
from lfd.datasets.disaster_tweets import make_cv_disaster_tweets
from lfd.layers import Attention
from lfd.utils.meters import AverageMeter, AccumulateMeter, get_scoring
from lfd.utils.pretrained import load_word_embedding


class DisasterTweetsDataset(Dataset):
    def __init__(self, words_or_texts, target=None):
        super(DisasterTweetsDataset, self).__init__()
        if isinstance(words_or_texts[0], (list, np.ndarray)):
            self.words = words_or_texts
        if isinstance(words_or_texts[0], str):
            pass
        self.target = target

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        if self.target is not None:
            return torch.LongTensor(self.words[item]), torch.FloatTensor([self.target[item]])
        return torch.LongTensor(self.words[item])


class DisasterTweetsBertDataset(Dataset):
    def __init__(self, words_or_texts, target, tokenizer, max_length=90):
        super(DisasterTweetsBertDataset, self).__init__()
        if isinstance(words_or_texts[0], (list, np.ndarray)):
            self.words = words_or_texts
        if isinstance(words_or_texts[0], str):
            encoding = tokenizer(words_or_texts, padding="max_length", truncation=True, max_length=max_length)
            self.words = encoding
        self.target = target

    def __len__(self):
        return len(self.words["input_ids"])

    def __getitem__(self, item):
        a, b = torch.LongTensor(self.words["input_ids"][item]), torch.LongTensor(self.words["attention_mask"][item])
        if self.target is not None:
            return a, b, torch.FloatTensor([self.target[item]])
        return a, b


class BasicLSTM(nn.Module):
    def __init__(self, max_word, max_seq_length, word_embedding=None, lstm_size=20, embedding_dim=50,
                 bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(max_word, embedding_dim=embedding_dim, padding_idx=0)
        if word_embedding is not None:
            self.embedding.weight.data = torch.FloatTensor(word_embedding)
            self.embedding.requires_grad_(False)
        self.lstm = nn.LSTM(embedding_dim, lstm_size, bidirectional=bidirectional, batch_first=True)
        self.mean_pool = lambda x: torch.mean(x, dim=1)
        self.max_pool = lambda x: torch.max(x, dim=1)[0]
        # self.attention_layer = Attention(lstm_size * (1 + int(bidirectional)), max_seq_length)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        # self.dropout = nn.Identity()
        self.head = nn.Linear(lstm_size * (1 + int(bidirectional)) * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        # print("embedding:", x.shape)
        x, _ = self.lstm(x)
        # print("lstm:", x.shape)
        mean_pooling = self.flatten(self.mean_pool(x))
        max_pooling = self.flatten(self.max_pool(x))
        # attention = self.attention_layer(x)
        x = torch.cat([mean_pooling, max_pooling], dim=1)
        # print("before head:", x.shape)
        x = self.dropout(x)
        x = self.head(x)
        # print("output:", x.shape)
        return x


def evaluate(model, dataloader, criterion, scoring, best_score=None, fold=1):
    model.eval()
    meter = AverageMeter()
    score_meter = AccumulateMeter(func=get_scoring(scoring))
    tqdm_obj = tqdm(dataloader, total=len(dataloader))
    with torch.no_grad():
        for *x, y in tqdm_obj:
            logit = model(*x)
            prediction = torch.sigmoid(logit)
            loss = criterion(logit, y)
            meter.update(loss.item(), len(y))
            score_meter.update(prediction.detach().cpu(), y.detach().cpu())
            tqdm_obj.set_postfix({"loss": f"{meter.avg:.5f}", scoring: f"{score_meter.avg:.5f}", "on": "'validation'",
                                  "best_score": f"{best_score:.5f}", "fold": fold})
    score = score_meter.avg
    if max(score, best_score) == score:
        tqdm_obj.set_postfix({"loss": f"{meter.avg:.5f}", scoring: f"{score_meter.avg:.5f}", "on": "'validation'",
                              "best_score": f"{max(score, best_score):.5f}(saved!)"})
        model_path = settings.MODELS / settings.NAME_DISASTER_TWEETS / f"{model.__class__.__name__}_{fold}.pth"
        print(f"save model to {model_path}")
        torch.save(model.state_dict(), model_path)
    return score


def predict(model, dataloader):
    model.eval()
    tqdm_obj = tqdm(dataloader, total=len(dataloader))
    predictions = []
    with torch.no_grad():
        for x in tqdm_obj:
            logit = model(x)
            prediction = torch.sigmoid(logit)
            tqdm_obj.set_postfix({"on": "'test'"})
            predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def train_one_epoch(model, dataloader, val_dataloader, criterion, optimizer, scoring, verbose=1, best_score=None,
                    label_smoothing: [int, float] = 0.0, fold=1, validation_on_step=0):
    model.train()
    meter = AverageMeter()
    score_meter = AccumulateMeter(func=get_scoring(scoring))
    tqdm_obj = tqdm(dataloader, total=len(dataloader), disable=not verbose)
    for step, (*x, y) in enumerate(tqdm_obj):
        logit = model(*x)
        if label_smoothing:
            target = y.clone()
            target[target == 0] = label_smoothing
            target[target == 1] = 1 - label_smoothing
            loss = criterion(logit, target)
        else:
            loss = criterion(logit, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), len(y))
        score_meter.update(torch.sigmoid(logit).detach().cpu(), y.detach().cpu())
        tqdm_obj.set_postfix({"loss": f"{meter.avg:.5f}", scoring: f"{score_meter.avg:.5f}", "on": "'train'",
                              "best": f"{best_score:.5f}", "fold": fold})
        if validation_on_step and (step + 1) % validation_on_step == 0:
            score = evaluate(model, val_dataloader, criterion, scoring, best_score=best_score, fold=fold)
            best_score = max(best_score, score)
            model.train()
    score = evaluate(model, val_dataloader, criterion, scoring, best_score=best_score, fold=fold)
    return max(score, best_score)


def fit(model_class, model_parameters, X, y, X_test, criterion, scoring="f1", cv=5, epochs=1, label_smoothing=0.1,
        validation_on_step=0):
    def _fit_one_fold(train_idx, val_idx, X, y, fold, epochs, label_smoothing=0.1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        dataset = DisasterTweetsDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        val_dataset = DisasterTweetsDataset(X_val, y_val)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
        model = model_class(**model_parameters)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_score = -float("inf")
        for epoch in range(epochs):
            score = train_one_epoch(model, dataloader, val_dataloader, criterion, optimizer, scoring=scoring, verbose=0,
                                    best_score=best_score, label_smoothing=label_smoothing, fold=fold,
                                    validation_on_step=validation_on_step)
            best_score = max(best_score, score)
        return best_score

    folds = make_cv_disaster_tweets(X, y, cv=cv)
    predictions = []
    scores = []
    oof = np.zeros((y.shape[0], ))
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        score = _fit_one_fold(train_idx, val_idx, X, y, fold, epochs=epochs, label_smoothing=label_smoothing)
        scores.append(score)
        model = model_class(**model_parameters)
        model_path = settings.MODELS / settings.NAME_DISASTER_TWEETS / f"{model.__class__.__name__}_{fold}.pth"
        print(f"load model from {model_path}")
        model.load_state_dict(torch.load(model_path))

        dataset = DisasterTweetsDataset(X_test, target=None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        test_pred = predict(model, dataloader)

        dataset = DisasterTweetsDataset(X[val_idx], target=None)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        val_pred = predict(model, dataloader)
        oof[val_idx] = val_pred.reshape(-1, )
        predictions.append(test_pred)
    print(f"mean: {np.mean(scores):.5f}, std: {np.std(scores):.5f}")
    return np.concatenate(predictions, axis=1).mean(axis=1), oof


if __name__ == '__main__':
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
    model_class = BasicLSTM
    model_parameters = dict(
        max_word=len(tokenizer.word_index) + 1, max_seq_length=33,
        lstm_size=40, embedding_dim=50, bidirectional=False
    )
    embedding_dim = 50
    embedding_path = settings.EMBEDDING_PATH / f"glove.6B.{embedding_dim}d.txt"
    print(f"load embedding from {embedding_path}")
    word_embedding = load_word_embedding(embedding_path, tokenizer.word_index,
                                         vector_size=embedding_dim)
    print(word_embedding.shape)
    model_parameters["word_embedding"] = word_embedding
    seed_everything(42)
    test_pred = fit(model_class, model_parameters, X, y, X_test, criterion, epochs=30, label_smoothing=0.1)
    df_submit = pd.DataFrame()
    df_submit["id"] = test["id"]
    df_submit["target"] = (test_pred > 0.5).astype(int)
    df_submit.to_csv(settings.OUTPUTS / settings.NAME_DISASTER_TWEETS / "lstm-ls.csv", index=False)
