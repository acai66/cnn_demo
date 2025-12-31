import torch
import os

from torch import nn
import torch.optim as optim

from tqdm import tqdm
from model import SimpleCNN
from dataset import get_data_loaders


def main():
    num_epochs = 500  # 训练轮数
    batch_size = 32  # 批次大小
    learning_rate = 0.001  # 学习率
    weight_decay = 1e-4  # 权重衰减
    num_classes = 2  # 类别数
    image_size = (32, 32)  # 图像大小
    data_dir = "./data"  # 数据目录
    checkpoint_dir = "./checkpoints"  # 模型保存目录

    print("加载数据集...")
    train_loader, test_loader = get_data_loaders(
        data_dir=data_dir, image_size=image_size, batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} 进行训练")

    print("初始化模型...")
    model = SimpleCNN(input_channels=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )  # 优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )  # 学习率调度器

    print("开始训练...")
    best_acc = 0.0
    os.makedirs(checkpoint_dir, exist_ok=True)  # 创建模型保存目录

    for epoch in range(1, num_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += inputs.shape[0]
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
        train_acc = train_correct / train_total
        train_loss /= len(train_loader)
        scheduler.step()

        # 测试
        model.eval()
        test_loss = 0.0
        test_total = 0
        test_correct = 0
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_total += inputs.shape[0]
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()
        test_acc = test_correct / test_total
        test_loss /= len(test_loader)
        print(
            f"轮次: {epoch}/{num_epochs} - "
            f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc * 100:.2f}% - "
            f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc * 100:.2f}%"
        )

        if test_acc >= best_acc:
            best_acc = test_acc
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model, checkpoint_path)
            print(f"保存最佳模型，准确率: {best_acc * 100:.2f}%")

    print(f"\n训练完成! 最佳准确率: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
