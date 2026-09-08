import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from iresnet import iresnet50  # backbone
from ArcFace_Loss import ArcFace  # ArcFace module


def main():
    # 1️⃣ Hyperparameters
    batch_size = 64
    lr = 0.1
    epochs = 20
    embedding_size = 512
    dataset_path = r'C:\Users\DucPc\Desktop\Dev\AI_Python\insightface\Demo_face_detection\dataset'

    # 2️⃣ Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()

    # 3️⃣ Data preprocessing and loading
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # fix normalize to 3 channels
    ])

    train_dataset = ImageFolder(root=dataset_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    num_classes = len(train_dataset.classes)

    # 4️⃣ Model & Loss
    model = iresnet50(num_features=embedding_size).to(device)
    metric_fc = ArcFace(embedding_size=embedding_size, num_classes=num_classes).to(device)

    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': metric_fc.parameters()}
    ], lr=lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 5️⃣ Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                embeddings = model(imgs)
                logits = metric_fc(embeddings, labels)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        scheduler.step()

    # 6️⃣ Save model weights
    torch.save(model.state_dict(), "iresnet_face.pth")
    torch.save(metric_fc.state_dict(), "arcface_fc.pth")


if __name__ == "__main__":
    main()
