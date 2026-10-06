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


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=4):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = stride == 1 and in_ch == out_ch

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            SqueezeExcitation(hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )
        self.features = nn.Sequential(
            MBConv(24, 24, expand_ratio=2),
            MBConv(24, 40, stride=2),
            MBConv(40, 40),
            MBConv(40, 80, stride=2),
            MBConv(80, 80),
            MBConv(80, 112, stride=2)
        )
        self.head = nn.Sequential(
            nn.Conv2d(112, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)

if __name__ == "__main__":
    run_training(EfficientNet, "EfficientNet")
