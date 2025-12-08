import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from model_utils import (set_seed, seed_worker, RealWorldDataset,
                        ASFDCLClassifier, compute_asf_dcl_losses)

# ------------------------------------------------------------------------------
# 1. Train / Evaluation
# ------------------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, device,
                lambda_dyn=0.1, lambda_flow=0.05,
                lambda_proto=0.1, lambda_contrast=0.15):
    model.train()
    total_loss = 0

    all_preds = []
    all_labels = []

    loss_accumulator = {
        "cls": 0.0,
        "dyn": 0.0,
        "flow_prior": 0.0,
        "proto": 0.0,
        "contrast": 0.0
    }

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits, details = model(x, return_details=True)
        loss, loss_dict = compute_asf_dcl_losses(
            logits, details, y,
            lambda_dyn=lambda_dyn,
            lambda_flow=lambda_flow,
            lambda_proto=lambda_proto,
            lambda_contrast=lambda_contrast,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k in loss_accumulator.keys():
            loss_accumulator[k] += loss_dict[k]

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    for k in loss_accumulator.keys():
        loss_accumulator[k] /= len(dataloader)

    return avg_loss, acc, f1, loss_accumulator


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, cm


# ------------------------------------------------------------------------------
# 2. Main 
# ------------------------------------------------------------------------------
def main():
    SEED = 42
    set_seed(SEED)

    DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/ASF/RealWorld2016-ChestOnly"
    SAVE_PATH = '/content/drive/MyDrive/Colab Notebooks/ASF/RealWorld2016-ChestOnly/RealWorld2016-ChestOnly_ASF.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    LAMBDA_DYN = 0.05
    LAMBDA_FLOW = 0.02
    LAMBDA_PROTO = 0.05
    LAMBDA_CONTRAST = 0.2

    train_dataset = RealWorldDataset(
        os.path.join(DATA_DIR, "X_train.npy"), 
        os.path.join(DATA_DIR, "y_train.npy")
    )
    test_dataset = RealWorldDataset(
        os.path.join(DATA_DIR, "X_test.npy"), 
        os.path.join(DATA_DIR, "y_test.npy")
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2,
                              worker_init_fn=seed_worker,
                              generator=g)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2,
                             worker_init_fn=seed_worker,
                             generator=g)
    
    model = ASFDCLClassifier(
        input_channels=6,
        latent_dim=64,
        hidden_dim=64,
        num_classes=8,
        num_heads=4,
        projection_dim=128
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_f1 = 0.0
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc, train_f1, loss_dict = train_epoch(
            model, train_loader, optimizer, DEVICE,
            lambda_dyn=LAMBDA_DYN,
            lambda_flow=LAMBDA_FLOW,
            lambda_proto=LAMBDA_PROTO,
            lambda_contrast=LAMBDA_CONTRAST,
        )

        test_loss, test_acc, test_f1, test_cm = evaluate(
            model, test_loader, DEVICE
        )

        scheduler.step()

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), SAVE_PATH)
            save_msg = "Saved Best Model"
        else:
            save_msg = ""

        epoch_time = time.time() - start_time
        log_msg = (f"[{epoch+1:02d}/{NUM_EPOCHS}] "
                   f"Train Loss: {train_loss:.3f} | F1: {train_f1:.4f}  |  "
                   f"Test F1: {test_f1:.4f} (Best: {best_f1:.4f}) {save_msg}")
        print(log_msg)

if __name__ == "__main__":
    main()