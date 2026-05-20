"""
Train the ActionLSTM on collected landmark sequences.

Usage:
    python -m training.train --data_dir data/landmarks --epochs 50 --lr 1e-3
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from pathlib import Path

from training.dataset import LandmarkDataset
from src.action_classifier import ActionLSTM, ACTION_LABELS


def train(data_dir: str, model_out: str = "models/action_lstm.pt",
          epochs: int = 50, lr: float = 1e-3, batch_size: int = 32,
          window_size: int = 20, device: str = "cpu"):

    dataset = LandmarkDataset(data_dir, window_size=window_size, augment=True)
    val_size = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ActionLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seqs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs = seqs.to(device)
                preds = model(seqs).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_out)
            print(f"  -> Saved best model ({val_acc:.3f})")

    print("\nClassification Report (best model):")
    model.load_state_dict(torch.load(model_out, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in val_loader:
            preds = model(seqs.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print(classification_report(all_labels, all_preds, target_names=ACTION_LABELS))
    print(f"\nBest val accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/landmarks")
    parser.add_argument("--model_out", default="models/action_lstm.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(**vars(args))
