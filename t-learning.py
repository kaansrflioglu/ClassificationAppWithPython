import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import json
from torchvision import models
from model import SimpleCNN

with open("config.json", "r") as config_file:
    config = json.load(config_file)

data_path_main = config["data_path_main"]
input_size = config["input-size"]
epochs = config["epochs"]
batch_size = config["batch-size"]
learning_rate = config["learning-rate"]
input_channels = config["inputChannels"]

pretrained_model = models.resnet18(pretrained=True)
SimpleCNN().load_state_dict(torch.load("model.pth"))

transfer_model = nn.Sequential(
    *list(pretrained_model.children())[:-1],
    nn.Flatten(),
    nn.Linear(pretrained_model.fc.in_features, SimpleCNN().fc2.in_features),
    nn.ReLU(),
    nn.Linear(SimpleCNN().fc2.in_features, input_channels)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transfer_model.to(device)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=data_path_main, transform=transform_train)
class_names = dataset.classes
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(transfer_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    transfer_model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = transfer_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    transfer_model.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = {class_name: 0 for class_name in class_names}
        class_total = {class_name: 0 for class_name in class_names}

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = transfer_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i, class_name in enumerate(class_names):
                class_labels = labels == i
                class_predicted = predicted[class_labels]
                class_total[class_name] += class_labels.sum().item()
                class_correct[class_name] += class_predicted.eq(i).sum().item()

        val_accuracy = 100 * correct / total
        save_file1 = "t-learning_training_results.txt"
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")
        with open(save_file1, "a") as file:
            file.write(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%\n")


overall_accuracy = 100 * correct / total

print("\nClass-wise Accuracy:\n")
for class_name in class_names:
    class_acc = 100 * class_correct[class_name] / class_total[class_name]
    class_acc_rounded = round(class_acc, 2)
    print(f"Accuracy of {class_name}: {class_acc_rounded}%")
print(f"\nOverall Accuracy: {round(overall_accuracy, 2)}%")

save_file2 = "t-learning_evaluation_results.txt"
with open(save_file2, "a") as file:
    file.write(f"Overall Accuracy: {round(overall_accuracy, 2)}%\n")
    file.write("\nClass-wise Accuracy:\n")
    for class_name in class_names:
        class_acc = 100 * class_correct[class_name] / class_total[class_name]
        class_acc_rounded = round(class_acc, 2)
        file.write(f"Accuracy of {class_name}: {class_acc_rounded}%\n")

torch.save(transfer_model.state_dict(), "t-model.pth")
