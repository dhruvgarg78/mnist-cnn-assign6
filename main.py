import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net

# Set seed for reproducibility
torch.manual_seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Datasets and loaders
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize model
model = Net().to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Params: {params}")

# Optimizer, scheduler, loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
criterion = nn.NLLLoss()

# Tracking
train_losses, val_losses, train_accs, val_accs = [], [], [], []

def train():
    model.train()
    correct = 0
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)
    train_losses.append(avg_loss)
    train_accs.append(acc)
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

def test():
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    val_losses.append(avg_loss)
    val_accs.append(acc)
    print(f"Val Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

# Run training loop and plot if executed directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    epochs = 19
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        train()
        test()
        scheduler.step()

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Loss over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy over Epochs")
    plt.show()
