import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support


class Transform(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y


def collate_fn(batch):
    N = len(batch)
    features, labels = zip(*batch)
    features = np.array(features)
    X = torch.Tensor(features).float()
    Y = torch.Tensor(labels).long()

    return X, Y


def create_dataloader(data,
                      shuffle=False,
                      batch_size=256):

    collate = lambda batch: collate_fn(batch)
    dataset = Transform(data)
    dataloader = DataLoader(dataset, batch_size,
                            shuffle, collate_fn=collate)
    return dataloader


def evaluate(model, dataloader, criterion, device):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)
            predictions.extend(logits.argmax(dim=1).cpu())
            labels.extend(Y.cpu())

    p, r, f1, _ = precision_recall_fscore_support(labels, predictions, 
                                                  average="binary", 
                                                  zero_division=0)
    return [p, r, f1]


def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()

    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()

    result = evaluate(model, dataloader, criterion, device)
    return result
