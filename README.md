# Adversarial Robustness — TML 2026, Assignment 3

Trains a robust image classifier (ResNet-18) using **PGD adversarial training** to defend against adversarial attacks, as part of the Trustworthy Machine Learning 2026 course (CISPA Helmholtz Center).

**Leaderboard score: 0.586** (clean accuracy ≈ 70%, robust accuracy ≈ 46%)

---

## How to Reproduce the Best Result

### 1. Install dependencies
```bash
pip install numpy torch torchvision
```

### 2. Download the dataset
Download `train.npz` from the [HuggingFace dataset](https://huggingface.co/datasets/SprintML/tml26_task3) and place it in the same folder as `train_robust_fixed.py`:

```bash
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='SprintML/tml26_task3', filename='train.npz', repo_type='dataset', local_dir='.')"
```

### 3. Run training
```bash
python3 train_robust_fixed.py
```

This trains for 60 epochs and saves the best checkpoint (by unified score) to `robust_model_fixed.pt`.

### 4. Submit
Edit `submission.py`:
```python
API_KEY    = "YOUR_API_KEY_HERE"
MODEL_PATH = "robust_model_fixed.pt"
MODEL_NAME = "resnet18"
SUBMIT     = True
```
Then run:
```bash
python3 submission.py
```

---

## Approach

**Method:** PGD adversarial training (Madry et al., 2018) — at every training step, adversarial examples are generated on-the-fly and the model is trained on those instead of clean images.

**Architecture:** Stock `torchvision.models.resnet18`, with only the final `fc` layer replaced to output 9 classes. (Modifying `conv1`/`maxpool` for CIFAR-style input causes a shape mismatch when the evaluation server loads the state dict.)

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Epochs | 60 |
| Batch size | 128 |
| Optimizer | SGD (momentum=0.9, nesterov, weight_decay=5e-4) |
| LR schedule | CosineAnnealingLR |
| PGD ε (train) | 8/255 |
| PGD α | 2/255 |
| PGD steps (train) | 7 |
| PGD steps (eval) | 20 |
| Label smoothing | 0.1 |
| Augmentation | Random crop (reflect pad) + horizontal flip |

The checkpoint with the best unified score (`0.5 × clean_accuracy + 0.5 × robust_accuracy`) on a held-out validation split (10% of training data) is saved.

---

## Files

- `train_robust_fixed.py` — training script
- `submission.py` — submits the trained model to the evaluation server
- `report.pdf` — full report (approach, results, conclusion)
