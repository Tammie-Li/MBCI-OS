"""
基于 4 个数据文件 (x_train, y_train, x_test, y_test) 的 EMGNet 训练示例。
功能：
- 加载 npy 数据并构建 DataLoader
- 简单 1D CNN（EMGNet 示例）
- 实时绘制训练/验证 loss 曲线
- 每个 epoch 更新混淆矩阵

依赖：torch、matplotlib、scikit-learn、numpy
"""

import argparse
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset


class EMGDataset(Dataset):
    """简单的 EMG 数据集封装。"""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert x.shape[0] == y.shape[0], "x 与 y 样本数需一致"
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EMGNet(nn.Module):
    """
    轻量 1D CNN 示例：
    - Conv -> BN -> ReLU -> Pool (重复)
    - 自适应池化保证不同长度可用
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # 输出 [B, 128, 1]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, C, T]
        x = self.features(x)
        x = self.classifier(x)
        return x


def setup_plots(num_classes: int, class_names):
    """初始化实时绘图（loss + 混淆矩阵）。"""
    plt.ion()
    fig, (ax_loss, ax_cm) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, ls="--", alpha=0.5)

    ax_cm.set_title("Confusion Matrix")
    im = ax_cm.imshow(np.zeros((num_classes, num_classes)), cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_xticks(range(num_classes))
    ax_cm.set_yticks(range(num_classes))
    ax_cm.set_xticklabels(class_names)
    ax_cm.set_yticklabels(class_names)
    plt.tight_layout()
    return fig, ax_loss, ax_cm, im


def update_plots(
    fig,
    ax_loss,
    ax_cm,
    im,
    train_losses,
    test_losses,
    cm: np.ndarray,
    class_names,
):
    """刷新图像：loss 曲线 + 混淆矩阵。"""
    ax_loss.cla()
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, ls="--", alpha=0.5)
    epochs = np.arange(1, len(train_losses) + 1)
    ax_loss.plot(epochs, train_losses, label="train", color="tab:blue")
    if test_losses:
        ax_loss.plot(epochs, test_losses, label="test", color="tab:orange")
    ax_loss.legend()

    ax_cm.cla()
    ax_cm.set_title("Confusion Matrix")
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(class_names)))
    ax_cm.set_yticks(range(len(class_names)))
    ax_cm.set_xticklabels(class_names)
    ax_cm.set_yticklabels(class_names)

    # 在格子里写数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)


def evaluate(model, loader, device) -> Tuple[float, float, np.ndarray]:
    """计算 loss、准确率和混淆矩阵。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])
    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    num_classes = model.classifier[-1].out_features
    cm = (
        confusion_matrix(labels_np, preds_np, labels=np.arange(num_classes))
        if labels_np.size > 0
        else np.zeros((num_classes, num_classes), dtype=int)
    )
    return avg_loss, acc, cm


def train(args):
    # 加载数据
    x_train = np.load(args.x_train)
    y_train = np.load(args.y_train)
    x_test = np.load(args.x_test)
    y_test = np.load(args.y_test)

    # 确保形状为 (N, C, T)
    if x_train.ndim != 3:
        raise ValueError("x_train 需为三维张量 (样本, 通道, 序列长度)")
    if x_test.ndim != 3:
        raise ValueError("x_test 需为三维张量 (样本, 通道, 序列长度)")

    num_classes = int(np.max([y_train.max(), y_test.max()])) + 1
    class_names = [str(i) for i in range(num_classes)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_ds = EMGDataset(x_train, y_train)
    test_ds = EMGDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EMGNet(in_channels=x_train.shape[1], num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    fig, ax_loss, ax_cm, im = setup_plots(num_classes, class_names)
    train_losses, test_losses = [], []
    best_acc = 0.0
    best_path = args.save_path

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        start = time.time()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total += x.size(0)

        avg_train_loss = running_loss / max(total, 1)
        train_losses.append(avg_train_loss)

        test_loss, test_acc, cm = evaluate(model, test_loader, device)
        test_losses.append(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model_state": model.state_dict(), "acc": best_acc}, best_path)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4f} | time={elapsed:.1f}s"
        )

        update_plots(fig, ax_loss, ax_cm, im, train_losses, test_losses, cm, class_names)

    print(f"训练完成，最佳准确率={best_acc:.4f}，模型已保存：{best_path}")
    plt.ioff()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="EMGNet 训练与实时可视化示例")
    parser.add_argument("--x-train", required=True, help="x_train.npy 路径")
    parser.add_argument("--y-train", required=True, help="y_train.npy 路径")
    parser.add_argument("--x-test", required=True, help="x_test.npy 路径")
    parser.add_argument("--y-test", required=True, help="y_test.npy 路径")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 线程数（Windows 建议 0）")
    parser.add_argument("--save-path", type=str, default="best_emgnet.pt", help="最佳模型保存路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

