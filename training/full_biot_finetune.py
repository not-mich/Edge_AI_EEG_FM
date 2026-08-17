"""
full_biot_finetune.py

- linear probing: BIOTEncoder fully frozen; trainable small classification head on central server

cmd line: python3 training/full_biot_finetune.py --data_root data/processed --pretrained_ckpt BIOT/pretrained-models/EEG-PREST-16-channels.ckpt
"""

import sys, os
import torch
import numpy as np
import pickle
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "BIOT"))

from model import BIOTClassifier

# load TUAB dataset
# BIOTClassifier
# load EEG-PREST-16-channels.ckpt into the encoder
# freeze model.biot
# Binary Cross-Entropy (BCE) loss (BCE vs label 0,1)
# Optimiser containing only training params
# Validation metrics
# fine-tuned checkpoint- classifier heard + trained classifier head best model saved

class TUABData(Dataset):
    def __init__(self, split_dir):
        self.files = [
            os.path.join(split_dir, f) 
            for f in os.listdir(split_dir) 
            if f.endswith(".pkl")
        ]

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            sample = pickle.load(f)

        X = sample["X"]
        # normalisation 
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        X = torch.tensor(X, dtype=torch.float32)  # eeg window size (16 channels, 2000 time pts)
        y = torch.tensor(sample["y"], dtype=torch.float32) # scalar 0.0 = normal, 1.0 = abnormal

        return X,y

    def __len__(self):
        return len(self.files)

# construct BIOT classifier, load encoder (pretrained weights) and freeze during training
def build_model(pretrained_ckpt, n_channels: int = 16, device: str = "cpu"):
    model = BIOTClassifier(
        emb_size = 256,
        heads = 8,
        depth = 4,
        n_classes = 1, # binary task
        n_fft = 200,
        hop_length = 100,
        n_channels = n_channels
    )

    # load encoder
    if pretrained_ckpt:
        state = torch.load(pretrained_ckpt, map_location = device)
        model.biot.load_state_dict(state)
        print(f"Loaded encoder (pretrained weights) from: {pretrained_ckpt}")

    # freeze model.biot weights during training
    for param in model.biot.parameters():
        param.requires_grad = False 
    
    return model.to(device)

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
        model.biot.eval() 
    else:
        model.eval()

    all_logits, all_labels = [], []
    total_loss = 0.0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            # forward pass
            # EEG window -> frozen encoder -> feature vector -> trainable classifier head -> one logit per sample
            logits = model(X).squeeze(1)
            loss = criterion(logits, y) # how wrong are the preds in this batch

            if train:
                optimizer.zero_grad()
                loss.backward() # compute gradients for only trainable head params
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())
    
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1/(1+np.exp(-all_logits)) # sigmoid: probability of "abnormal"
    preds = (probs >= 0.5).astype(int) # threshold probability into a hard 0/1 prediction

    # print metrics
    loss = total_loss/len(loader.dataset)
    balanced_acc = balanced_accuracy_score(all_labels, preds)
    auroc = roc_auc_score(all_labels, probs)
    auprc = average_precision_score(all_labels, probs)

    return loss, balanced_acc, auroc, auprc

def main():
    # parse cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pretrained_ckpt", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", default="biot_finetuned_best_ckpt.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # build datasets/loader for train and val splits
    # training data is shuffled each epoch; val is not
    train_dataset = TUABData(os.path.join(args.data_root, "train"))
    val_dataset = TUABData(os.path.join(args.data_root, "val"))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model build
    model = build_model(
        args.pretrained_ckpt,
        n_channels=16,
        device=device
    )

    # verify freeze
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} total (frozen encoder)")

    # optimiser build from trainable_params
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = -1.0 

    for epoch in range(1, args.epochs + 1):
        # one training pass (updates head weights) + one val pass (no updates)
        train_loss, train_balanced_acc, train_auroc, train_auprc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_balanced_acc, val_auroc, val_auprc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"\nEpoch {epoch}:")
        print(f"train loss = {train_loss:.3f}")
        print(f"val loss = {val_loss:.3f}")
        print(f"balanced accuracy score = {val_balanced_acc:.3f}")
        print(f"auroc = {val_auroc:.3f}")
        print(f"average precision score = {val_auprc:.3f}")

        # save if val AUROC improved
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), args.save_path)
            print(f"saved new best checkpoint in {args.save_path}")

    print(f"Best validation AUROC score: {best_auroc:.3f}, Checkpoint: {args.save_path}")

if __name__ == "__main__":
    main()