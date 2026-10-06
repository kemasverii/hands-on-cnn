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


class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth_rate, 3, padding=1, bias=False)
        )

    def forward(self, x):
        new_features = self.block(x)
        return torch.cat([x, new_features], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth_rate, num_layers):
        super().__init__()
        layers = []
        channels = in_ch

        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.block(x)

class DenseNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.db1 = DenseBlock(32, growth_rate=16, num_layers=4)
        self.tr1 = Transition(self.db1.out_channels, 48)

        self.db2 = DenseBlock(48, growth_rate=16, num_layers=4)
        self.tr2 = Transition(self.db2.out_channels, 64)

        self.db3 = DenseBlock(64, growth_rate=16, num_layers=4)

        self.head = nn.Sequential(
            nn.BatchNorm2d(self.db3.out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.db3.out_channels, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.tr1(self.db1(x))
        x = self.tr2(self.db2(x))
        x = self.db3(x)
        return self.head(x)

if __name__ == "__main__":
    run_training(DenseNet, "DenseNet")
