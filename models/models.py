import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def load_and_preprocess_data(price_path="archive/prices-split-adjusted.csv", sentiment_path="sentiment_scores.csv"):
    df = pd.read_csv(price_path)
    df["date"] = pd.to_datetime(df["date"])

    sentiment_df = pd.read_csv(sentiment_path)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    df = pd.merge(df, sentiment_df, on="date", how="left")
    df["sentiment"] = df["sentiment"].fillna(0)

    df["50_MA"] = df["close"].rolling(window=50).mean()
    df["200_MA"] = df["close"].rolling(window=200).mean()
    df["volatility"] = df["close"].rolling(window=20).std()
    df["z_score"] = (df["close"] - df["50_MA"]) / df["volatility"]
    df.dropna(inplace=True)
    df["target"] = (abs(df["z_score"]) > 2).astype(int)

    return df

def create_sequences(data, sequence_length=30):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        features = data.iloc[i:i + sequence_length][['close', '50_MA', '200_MA', 'volatility', 'sentiment']].values
        label = data.iloc[i + sequence_length]['target']
        X.append(features)
        y.append(label)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return model, history


def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy
