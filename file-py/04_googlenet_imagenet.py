import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
TRAIN_SAMPLES = 4000
VAL_SAMPLES = 800
IMAGENET_ROOT = "./imagenet"

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloaders():
    train_dir = Path(IMAGENET_ROOT) / "train"
    val_dir = Path(IMAGENET_ROOT) / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Folder ImageNet tidak ditemukan. Gunakan struktur:\n"
            "./imagenet/train/<nama_kelas>/*.jpg\n"
            "./imagenet/val/<nama_kelas>/*.jpg"
        )

    transform_train = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    train_set = datasets.ImageFolder(train_dir, transform=transform_train)
    val_set = datasets.ImageFolder(val_dir, transform=transform_val)

    if train_set.classes != val_set.classes:
        raise ValueError("Class folder pada train dan val harus sama.")

    g = torch.Generator().manual_seed(SEED)
    train_n = min(TRAIN_SAMPLES, len(train_set))
    val_n = min(VAL_SAMPLES, len(val_set))

    train_idx = torch.randperm(len(train_set), generator=g)[:train_n]
    val_idx = torch.randperm(len(val_set), generator=g)[:val_n]

    train_loader = DataLoader(
        Subset(train_set, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        Subset(val_set, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return train_loader, val_loader, len(train_set.classes)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def run_training(model, model_name):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = get_dataloaders()

    model = model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model: {model_name}")
    print("Dataset: ImageNet subset")
    print(f"Classes: {num_classes}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, c1, c3r, c3, c5r, c5, pool_proj):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 1),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, c3r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3r, c3, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, c5r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5r, c5, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.inc1 = InceptionBlock(64, 24, 24, 32, 8, 8, 16)
        self.inc2 = InceptionBlock(80, 32, 32, 48, 8, 16, 16)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.inc3 = InceptionBlock(112, 48, 32, 48, 8, 16, 16)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool(x)
        x = self.inc3(x)
        return self.head(x)

if __name__ == "__main__":
    run_training(GoogLeNet, "GoogLeNet")
