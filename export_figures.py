#!/usr/bin/env python3
"""Regenera las figuras del informe y las guarda en images/ (misma lógica que Tarea2.ipynb)."""
from __future__ import annotations

import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

os.makedirs("images", exist_ok=True)

df = pd.read_csv("winequality-red.csv")
corr = df.corr(numeric_only=True)
corr_target = corr["quality"].drop("quality").sort_values(key=np.abs, ascending=False)
ordered_features = corr_target.index.tolist()
selected = []
max_features = 6
for feat in ordered_features:
    if len(selected) == max_features:
        break
    if not selected:
        selected.append(feat)
    else:
        too_correlated = False
        for s in selected:
            if abs(corr.loc[feat, s]) > 0.8:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(feat)

selected_features = selected

fig, ax = plt.subplots(figsize=(10, 8))
c = corr.values
im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
labels = list(corr.columns)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{c[i, j]:.3f}", ha="center", va="center", fontsize=6)
ax.set_title("Matriz de correlación (variables numéricas)")
plt.tight_layout()
plt.savefig("images/fig_corr_heatmap.png", dpi=150)
plt.close()

target = "quality"
for col in selected_features:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[col], df[target], s=10, alpha=0.6)
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(f"{col} vs {target}")
    plt.tight_layout()
    if col == "alcohol":
        plt.savefig("images/alcohol_vs_quality.png", dpi=150)
    elif col == "volatile acidity":
        plt.savefig("images/volatile_acidity_vs_quality.png", dpi=150)
    plt.close()

for col in selected_features:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col], vert=True)
    plt.title(f"Boxplot de {col}")
    plt.xlabel(col)
    plt.tight_layout()
    if col == "sulphates":
        plt.savefig("images/boxplot_sulfate.png", dpi=150)
    elif col == "density":
        plt.savefig("images/boxplot_density.png", dpi=150)
    plt.close()

for col in selected_features:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=30, edgecolor="black", alpha=0.7)
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    if col == "alcohol":
        plt.savefig("images/hist_alcohol.png", dpi=150)
    elif col == "volatile acidity":
        plt.savefig("images/hist_volatile_acidity.png", dpi=150)
    plt.close()

df_clean = df.copy()
for col in selected_features:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

df = df_clean.reset_index(drop=True)
threshold = 7
df["quality_binary"] = (df["quality"] >= threshold).astype(int)

X = df[selected_features].copy()
y = df["quality_binary"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
y_val_t = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)
y_test_t = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)
input_dim = X_train_t.shape[1]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_one_run(lr, batch_size, epochs, verbose=False):
    model = LogisticRegressionModel(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits_val = model(X_val_t)
        probs_val = torch.sigmoid(logits_val).cpu().numpy().ravel()
    y_val_pred = (probs_val >= 0.5).astype(int)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    val_rec = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, probs_val)
    metrics = {
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "val_accuracy": val_acc,
        "val_precision": val_prec,
        "val_recall": val_rec,
        "val_f1": val_f1,
        "val_auc": val_auc,
    }
    history = {"train_losses": train_losses, "val_losses": val_losses}
    return model, metrics, history


experiments = [
    {"lr": 1e-4, "batch_size": 32, "epochs": 50},
    {"lr": 3e-4, "batch_size": 32, "epochs": 50},
    {"lr": 1e-3, "batch_size": 32, "epochs": 50},
    {"lr": 3e-3, "batch_size": 32, "epochs": 50},
    {"lr": 1e-3, "batch_size": 64, "epochs": 50},
    {"lr": 3e-3, "batch_size": 64, "epochs": 50},
    {"lr": 1e-3, "batch_size": 128, "epochs": 50},
    {"lr": 3e-3, "batch_size": 128, "epochs": 50},
    {"lr": 1e-3, "batch_size": 64, "epochs": 100},
    {"lr": 3e-3, "batch_size": 64, "epochs": 100},
]

results = []
histories = []
best_model = None
best_f1 = -1.0

for i, cfg in enumerate(experiments, start=1):
    model, metrics, history = train_one_run(**cfg)
    results.append(metrics)
    histories.append(history)
    if metrics["val_f1"] > best_f1:
        best_f1 = metrics["val_f1"]
        best_model = model

best_by_f1 = sorted(results, key=lambda m: m["val_f1"], reverse=True)[0]

for i, (cfg, history) in enumerate(zip(experiments, histories), start=1):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_losses"], label="Training")
    plt.plot(history["val_losses"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Experimento {i}: lr={cfg['lr']}, batch={cfg['batch_size']}, epochs={cfg['epochs']}")
    plt.legend()
    plt.tight_layout()
    if i == 5:
        plt.savefig("images/exp5_training_validation.png", dpi=150)
    if i == 10:
        plt.savefig("images/exp10_training_validation.png", dpi=150)
    plt.close()

best_model.eval()
with torch.no_grad():
    logits_test = best_model(X_test_t)
    probs_test = torch.sigmoid(logits_test).cpu().numpy().ravel()
y_test_pred = (probs_test >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="Blues")
plt.title("Matriz de confusión - conjunto de test")
plt.tight_layout()
plt.savefig("images/matriz_de_confusion.png", dpi=150)
plt.close()

print("Figuras guardadas en images/. Mejor F1 (validación):", best_by_f1["val_f1"])
