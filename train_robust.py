"""
Adversarial Training Script - Assignment 3: Robustness
Uses STOCK torchvision ResNet (7x7 conv1 + maxpool). Only the final fc layer is replaced for 9 classes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.models as models
import random
import os

# ── Hyperparameters ────────────────────────────────────────────────────────────
ARCH        = "resnet18"
SAVE_PATH   = "robust_model_fixed.pt"
NUM_CLASSES = 9
EPOCHS      = 60
BATCH_SIZE  = 128
LR          = 0.1
WEIGHT_DECAY= 5e-4
MOMENTUM    = 0.9

EPS         = 8  / 255
ALPHA       = 2  / 255
PGD_STEPS   = 7
EVAL_STEPS  = 20

VAL_FRAC    = 0.1
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
print(f"Using device: {device}")


def find_data_file():
    for p in ["train.npz", "./train.npz", os.path.join(os.path.dirname(__file__), "train.npz")]:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("train.npz not found! Please place it in the same folder as this script.")


def load_data(path):
    data = np.load(path)
    images = data["images"].astype(np.float32) / 255.0
    labels = data["labels"].astype(np.int64)
    if images.ndim == 4 and images.shape[-1] == 3:
        images = images.transpose(0, 3, 1, 2)
    return torch.from_numpy(images), torch.from_numpy(labels)


def augment_batch(x):
    if random.random() > 0.5:
        x = torch.flip(x, dims=[3])
    pad = 4
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    _, _, h, w = x.shape
    top  = random.randint(0, h - 32)
    left = random.randint(0, w - 32)
    x = x[:, :, top:top+32, left:left+32]
    return x


# ── Model: STOCK torchvision ResNet, only fc replaced ──────────────────────────
def build_model(arch):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet34":
        model = models.resnet34(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # ONLY replace the final layer - keep stock conv1/maxpool
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def pgd_attack(model, x, y, eps, alpha, steps, random_start=True):
    model.eval()
    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.clamp(x_adv, x - eps, x + eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = augment_batch(x)
        x_adv = pgd_attack(model, x, y, EPS, ALPHA, PGD_STEPS)
        model.train()
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_clean(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def evaluate_robust(model, loader, eps, alpha, steps):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_attack(model, x, y, eps, alpha, steps)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    data_path = find_data_file()
    print(f"Loading data from: {data_path}")
    images, labels = load_data(data_path)
    dataset = TensorDataset(images, labels)

    val_size   = int(len(dataset) * VAL_FRAC)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = build_model(ARCH)
    print(f"Architecture: {ARCH} (stock, fc->9) | Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_score = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        if epoch % 5 == 0 or epoch == EPOCHS:
            clean_acc  = evaluate_clean(model, val_loader)
            robust_acc = evaluate_robust(model, val_loader, EPS, ALPHA, EVAL_STEPS)
            score      = 0.5 * clean_acc + 0.5 * robust_acc

            print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {train_loss:.4f} | "
                  f"train_acc: {train_acc:.3f} | clean: {clean_acc:.3f} | "
                  f"robust: {robust_acc:.3f} | score: {score:.3f} | "
                  f"lr: {scheduler.get_last_lr()[0]:.5f}")

            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"  ✓ New best score {best_score:.4f} — model saved to {SAVE_PATH}")
        else:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {train_loss:.4f} | "
                  f"train_acc: {train_acc:.3f} | lr: {scheduler.get_last_lr()[0]:.5f}")

    print(f"\nTraining complete. Best unified score: {best_score:.4f}")
    print(f"Model saved to: {SAVE_PATH}")

    # sanity check
    print("\nRunning submission sanity check...")
    check_model = build_model(ARCH)
    check_model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    check_model.eval()
    dummy = torch.zeros(1, 3, 32, 32).to(device)
    out   = check_model(dummy)
    assert out.shape == (1, NUM_CLASSES), f"Bad output shape: {out.shape}"
    print(f"  ✓ Output shape correct: {out.shape}")
    print("  ✓ Ready to submit!")


if __name__ == "__main__":
    main()
