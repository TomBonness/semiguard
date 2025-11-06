import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os

from preprocessing import load_and_clean, prepare_features, split_data
from model import DefectClassifier


def make_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader


def train():
    # load and prep data
    df = load_and_clean('../data/uci-secom.csv')
    X, y, scaler = prepare_features(df, scaler_path='../models/scaler.pkl')
    X_train, X_test, y_train, y_test = split_data(X, y)

    n_features = X_train.shape[1]
    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test)

    # weight the loss to handle class imbalance
    # pos_weight = num_negative / num_positive
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    pos_weight = torch.tensor([n_neg / n_pos])
    print(f"\npos_weight: {pos_weight.item():.2f} (compensating for {n_neg}:{n_pos} imbalance)")

    model = DefectClassifier(n_features)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    # save everything
    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), '../models/defect_model.pt')
    print("\nModel saved to models/defect_model.pt")

    metadata = {
        'n_features': n_features,
        'epochs': epochs,
        'learning_rate': 0.001,
        'pos_weight': pos_weight.item(),
        'train_samples': len(y_train),
        'test_samples': len(y_test)
    }
    with open('../models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to models/metadata.json")

    return model, test_loader, y_test


if __name__ == '__main__':
    model, test_loader, y_test = train()
