import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import SimpleCNN
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)
epochs = config["epochs"]


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if labels.size(0) != outputs.size(0):
            labels = labels[:outputs.size(0)]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy


input_size = config["input-size"]
batchSize = config["batch-size"]


def train_main(data_path):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = config["batch-size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(num_classes=len(dataset.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    learning_rate = config["learning-rate"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    save_file = "training_results.txt"
    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")
        with open(save_file, "a") as file:
            file.write(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%\n")
    torch.save(model.state_dict(), "model.pth")


def run_train_process(data_path):
    train_main(data_path)
